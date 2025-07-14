import { McpAgent } from "agents/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { Hono } from "hono";
import * as jose from "jose";

interface Env {
  JWT_SECRET: string;
  AI: Ai;
  VECTORIZE: Vectorize;
  DB: D1Database;
}

type Scope = "default" | "project";

type Props = {
  bearerToken: string;
  payload: Payload;
  isProjectScoped: boolean;
};

type Payload = {
  user_id: string;
  project_id: string;
};
const app = new Hono<{
  Bindings: Env;
}>();

type State = null;

export class LumiLinkMCP extends McpAgent<Env, State, Props> {
  server = new McpServer({
    name: "LumiLink brain core",
    version: "1.0.0",
  });

  async init() {
    this.server.tool(
      "retrieveKnowledgeBase",
      "retrieves relevant knowledgebase of user in LumiLink system",
      {
        query: z
          .string()
          .describe(
            "a query to search for relevant information in the knowledge base"
          ),
      },
      async ({ query }) => {
        let queryEmbedding;
        let scope: Scope = "default";

        try {
          const response = await this.env.AI.run("@cf/baai/bge-base-en-v1.5", {
            text: query,
          });
          queryEmbedding = response.data[0];
        } catch (error) {
          console.error("Error generating embedding:", error);
          return {
            content: [{ type: "text", text: "Embedding generation failed." }],
          };
        }

        if (!queryEmbedding) {
          return {
            content: [
              { type: "text", text: "No embedding returned from model." },
            ],
          };
        }

        const vectorFilter: Record<string, any> = {
          userId: this.props.payload.user_id,
        };

        // Handle project scoped case
        if (this.props.isProjectScoped) {
          scope = "project";
          const kbs = await this.env.DB.prepare(
            `SELECT kb.id
			   FROM KnowledgeBase kb
			   JOIN ProjectKnowledgeBase pkb ON pkb.knowledgeBaseId = kb.id
			   WHERE pkb.projectId = ?
				 AND kb.scope = 'project';`
          )
            .bind(this.props.payload.project_id)
            .all();

          const kbIds = kbs.results.map((kb) => kb.id);

          if (kbIds.length === 0) {
            return {
              content: [
                { type: "text", text: "No knowledge bases found for project." },
              ],
            };
          }

          vectorFilter.knowledgeBaseId = { $in: kbIds };
        }

        let vectorizeResults;

        try {
          vectorizeResults = await this.env.VECTORIZE.query(queryEmbedding, {
            namespace: "knowledge_base",
            topK: 15,
            returnMetadata: true,
            filter: vectorFilter,
          });
        } catch (err) {
          console.error("Vector search error:", err);
          return {
            content: [{ type: "text", text: "Vector search failed." }],
          };
        }

        const matches = Array.isArray(vectorizeResults?.matches)
          ? vectorizeResults.matches
          : [];

        if (matches.length === 0) {
          return {
            content: [
              { type: "text", text: "No matching knowledge chunks found." },
            ],
          };
        }

        const numericIds = matches
          .map((match) => parseInt(match.id.split("-")[1]))
          .filter((id) => !isNaN(id));

        if (numericIds.length === 0) {
          return {
            content: [
              { type: "text", text: "No valid knowledge chunk IDs found." },
            ],
          };
        }

        const dbQuery = `
			SELECT 
			  kc.*, 
			  d.name AS dataset_name, 
			  kb.name AS knowledge_base_name
			FROM 
			  knowledgeChunk kc
			JOIN 
			  dataset d ON kc.datasetId = d.id
			JOIN 
			  knowledgeBase kb ON d.knowledgeBaseId = kb.id
			WHERE 
			  kc.id IN (${numericIds.map(() => "?").join(",")})
			  AND d.enable = 1
			  AND kc.isDeleted = 0
			  AND kb.scope = ?;
		  `;

        const result = await this.env.DB.prepare(dbQuery)
          .bind(...numericIds, scope)
          .all();

        const knowledgeChunks = result.results as Array<{
          id: number;
          content: string;
          dataset_name: string;
          knowledge_base_name: string;
        }>;

        if (knowledgeChunks.length === 0) {
          return {
            content: [
              {
                type: "text",
                text: "No relevant knowledge found for this query.",
              },
            ],
          };
        }

        const scoreMap = new Map(
          matches.map((match) => [
            parseInt(match.id.split("-")[1]),
            match.score,
          ])
        );

        const knowledgeContext = `\n<KNOWLEDGE_BASE>\n${knowledgeChunks
          .map((chunk, index) => {
            const score = scoreMap.get(chunk.id) || 0;
            const source = `${chunk.knowledge_base_name} > ${chunk.dataset_name}`;
            return `## Section ${
              index + 1
            }: ${source} (Similarity Score: ${score.toFixed(3)})\n${
              chunk.content
            }\n`;
          })
          .join("\n---\n")}\n</KNOWLEDGE_BASE>`;

        return {
          content: [{ type: "text", text: knowledgeContext }],
        };
      }
    );
  }
}

app.mount("/", async (req, env, ctx) => {
  const authHeader = req.headers.get("authorization");
  if (!authHeader) {
    return new Response("Unauthorized", { status: 401 });
  }

  try {
    const token = authHeader.replace("Bearer ", "");
    console.log("token", token);
    console.log("env.JWT_SECRET", env.JWT_SECRET);
    const secret = new TextEncoder().encode(env.JWT_SECRET);
    console.log("secret", secret);
    const { payload }: { payload: Payload } = await jose.jwtVerify(
      token,
      secret
    );
    console.log("payload", payload);
    // const fakePayload = {
    //   user_id: "15",
    //   project_id: "3",
    // };

    ctx.props = {
      bearerToken: authHeader,
      payload: payload,
      isProjectScoped: !!payload.project_id,
    };
  } catch (err) {
    console.log("error", err);
    return new Response("Unauthorized", { status: 401 });
  }
  return LumiLinkMCP.mount("/sse").fetch(req, env, ctx);
});

export default app;
