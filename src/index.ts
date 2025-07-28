import { McpAgent } from "agents/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { Hono } from "hono";
import * as jose from "jose";
import {
  ENTITY_PRIORITY,
  PaginatedResult,
  PaginationParams,
  UserEntityWithId,
} from "./type";

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
  agent_id: string;
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
			  kb.name AS knowledge_base_name,
			  kb.id AS knowledge_base_id,
			  d.id AS dataset_id
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
          knowledge_base_id: number;
          dataset_id: number;
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
        const generateSourceUrl = (datasetId: number, kbId: number) => {
          return `https://lumilink.defikit.net/knowledge/dataset/chunk?id=${datasetId}&kbId=${kbId}`;
        };
        const knowledgeContext = `\n<KNOWLEDGE_BASE>\n${knowledgeChunks
          .map((chunk, index) => {
            const score = scoreMap.get(chunk.id) || 0;
            const source = `${chunk.knowledge_base_name} > ${chunk.dataset_name}`;
            const sourceUrl = generateSourceUrl(
              chunk.dataset_id,
              chunk.knowledge_base_id
            );
            return `## Section ${
              index + 1
            }: ${source} (Similarity Score: ${score.toFixed(3)})\n${
              chunk.content
            }\n[Source Name: ${chunk.dataset_name} Source URL: ${sourceUrl}]`;
          })
          .join("\n---\n")}\n</KNOWLEDGE_BASE>`;

        return {
          content: [{ type: "text", text: knowledgeContext }],
        };
      }
    );
    this.server.tool(
      `getUserPersonalizedInfo`,
      `Get user's personalized information including memories,entities, characteristics, and context to enhance system prompts`,
      {
        query: z
          .string()
          .describe(
            "a query to search for relevant information in the user's personalized information"
          ),
      },
      async ({ query }) => {
        const userEntitiesResult = await this.getAgentEntities(
          this.props.payload.user_id,
          this.props.payload.agent_id
        );
        const userEntities = Array.isArray(userEntitiesResult)
          ? userEntitiesResult
          : userEntitiesResult.data;

        const [
          conversationMemories,
          interactionMemories,
          userEntitiesFormatted,
        ] = await Promise.all([
          this.getConversationMemories(
            query,
            this.props.payload.user_id,
            this.props.payload.project_id,
            this.props.payload.agent_id
          ),
          this.getInteractionMemories(
            query,
            this.props.payload.user_id,
            this.props.payload.project_id
          ),
          this.formatEntitiesForPrompt(
            userEntities,
            this.props.payload.agent_id,
            this.props.payload.project_id
              ? parseInt(this.props.payload.project_id)
              : undefined
          ),
        ]);

        // Build sections dynamically
        const sections = [];

        if (userEntitiesFormatted) {
          sections.push(userEntitiesFormatted.trim());
        }

        if (conversationMemories) {
          sections.push(conversationMemories.trim());
        }

        if (interactionMemories) {
          sections.push(interactionMemories.trim());
        }

        if (sections.length === 0) {
          return {
            content: [
              {
                type: "text",
                text: "No personalized information available for this query.",
              },
            ],
          };
        }

        const userPersonalizedInfo = `
## PERSONALIZED CONTEXT

${sections.join("\n\n")}

---

## USAGE GUIDELINES:
- **Follow specific instructions** provided for each section above
- **Integrate naturally** - blend personalized information seamlessly into responses
- **Prioritize relevance** - only use information that adds value to the conversation
- **Maintain authenticity** while being contextually aware`;

        return {
          content: [{ type: "text", text: userPersonalizedInfo }],
        };
      }
    );
  }

  private async getConversationMemories(
    query: string,
    userId: string,
    projectId?: string,
    agentId?: string,
    namespace: string = `memories-${userId}`
  ): Promise<string> {
    try {
      // Generate embedding for the query
      const response = await this.env.AI.run("@cf/baai/bge-base-en-v1.5", {
        text: query,
      });
      const queryEmbedding = response.data[0];

      if (!queryEmbedding) return "";

      // Set up vector filter (replicates getMemories function logic)
      const vectorFilter: Record<string, any> = {
        userId: userId,
      };

      // Add project filtering if in project scope
      if (projectId && this.props.isProjectScoped) {
        vectorFilter.projectId = projectId;
      }

      // Add agent filtering if provided
      // if (agentId) {
      //   vectorFilter.agentId = agentId;
      // }

      console.log("vectorFilter", vectorFilter);
      console.log("namespace", namespace);
      // Query vectorize for memories
      const vectorizeResults = await this.env.VECTORIZE.query(queryEmbedding, {
        namespace: namespace,
        returnMetadata: true,
        filter: vectorFilter,
      });

      const matches = Array.isArray(vectorizeResults?.matches)
        ? vectorizeResults.matches
        : [];

      console.log("matches", matches);

      if (matches.length === 0) return "";

      console.log("matches", matches);

      // Filter and format memories (replicates commonMemories logic)
      const relevantMemories = matches
        .filter((match) => match.score > 0.6) // Only include high-relevance memories
        .slice(0, 500) // Limit to top 500 memories
        .map((match, index) => {
          const memoryText = match.metadata?.text || "No content available";
          const score = match.score.toFixed(3);
          return `${index + 1}. ${memoryText} (Relevance: ${score})`;
        })
        .join("\n");

      console.log("relevantMemories", relevantMemories);

      if (relevantMemories) {
        return `### CONVERSATION MEMORIES:

${relevantMemories}

**Instructions:**
- Reference past conversations when relevant to current discussion
- Use these memories to provide more personalized and contextual responses
- Don't force connections if they don't naturally fit the conversation`;
      }

      return "";
    } catch (error) {
      console.error("Error retrieving conversation memories:", error);
      return "";
    }
  }

  private async getInteractionMemories(
    query: string,
    userId: string,
    projectId?: string
  ): Promise<string> {
    try {
      // Get social interactions for the user (replicates applyMemoriesFromInteractions logic)
      let interactionQuery = `SELECT socialId FROM socialInteraction 
                             WHERE userId = ?`;
      const params = [userId];

      if (projectId && this.props.isProjectScoped) {
        interactionQuery += ` AND projectId = ?`;
        params.push(projectId);
      }

      interactionQuery += ` ORDER BY createdDate DESC`;

      const interactions = await this.env.DB.prepare(interactionQuery)
        .bind(...params)
        .all();

      if (interactions.results.length === 0) {
        console.log(
          "getInteractionMemories: no interactions found. Skipping memory retrieval."
        );
        return "";
      }

      const socialIds = interactions.results.map((row: any) => row.socialId);

      // Get interaction details
      const interactionDetails = await this.env.DB.prepare(
        `SELECT socialId, threadId FROM socialInteractionDetail 
         WHERE socialId IN (${socialIds.map(() => "?").join(",")})`
      )
        .bind(...socialIds)
        .all();

      // Generate embedding for query
      const response = await this.env.AI.run("@cf/baai/bge-base-en-v1.5", {
        text: query,
      });
      const queryEmbedding = response.data[0];

      if (!queryEmbedding) return "";

      let allMemories: any[] = [];

      // Query each namespace for memories (replicates the Promise.all logic)
      const memoryPromises = interactionDetails.results.map(
        async (detail: any) => {
          const namespace = `telegram-${detail.socialId}-${detail.threadId}`;

          try {
            const vectorizeResults = await this.env.VECTORIZE.query(
              queryEmbedding,
              {
                namespace: namespace,
                returnMetadata: true,
                filter: { userId: userId },
              }
            );

            return vectorizeResults?.matches || [];
          } catch (err) {
            console.warn(`Failed to query namespace ${namespace}:`, err);
            return [];
          }
        }
      );

      const memoryResults = await Promise.all(memoryPromises);

      // Combine all memories
      for (const memories of memoryResults) {
        if (memories && memories.length > 0) {
          allMemories = allMemories.concat(memories);
        }
      }

      if (allMemories.length === 0) return "";

      // Format memories (replicates the formatting logic)
      const relevantMemories = allMemories
        .filter((match) => match.score > 0.6) // Only include high-relevance memories
        .sort((a, b) => b.score - a.score) // Sort by relevance score descending
        .slice(0, 500) // Limit to top 500 memories
        .map((match, index) => {
          const memoryText = match.metadata?.text || "No content available";
          const score = match.score.toFixed(3);
          const source = match.metadata?.namespace || "Unknown source";
          return `${
            index + 1
          }. ${memoryText} (Relevance: ${score}, Source: ${source})`;
        })
        .join("\n");

      if (relevantMemories) {
        return `### SOCIAL INTERACTION MEMORIES:

${relevantMemories}

**Instructions:**
- Use these memories from past social interactions to provide contextual responses
- Reference previous group conversations and social contexts when appropriate
- Maintain conversational continuity based on past interactions`;
      }

      return "";
    } catch (error) {
      console.error("Error retrieving interaction memories:", error);
      return "";
    }
  }

  // Step 1: Get agent entities (replicates userEntityService.getAgentEntities)
  private async getAgentEntities(
    userId: string,
    agentId?: string,
    pagination?: PaginationParams
  ): Promise<UserEntityWithId[] | PaginatedResult<UserEntityWithId>> {
    try {
      if (!agentId) {
        return {
          data: [],
          pagination: {
            page: 1,
            limit: 20,
            total: 0,
            totalPages: 0,
            hasNextPage: false,
            hasPrevPage: false,
          },
        };
      }
      const page = Math.max(1, pagination?.page || 1);
      const limit = Math.min(100, Math.max(1, pagination?.limit || 20));
      const offset = (page - 1) * limit;

      const totalResult = await this.env.DB.prepare(
        `SELECT COUNT(*) as count
         FROM UserEntity
         WHERE userId = ? AND agentId = ? AND projectId IS NULL`
      )
        .bind(userId, agentId)
        .first();

      const total = (totalResult?.count as number) || 0;
      const totalPages = Math.ceil(total / limit);

      // Main query (enrich với DynamicEntityType)
      const entities = await this.env.DB.prepare(
        `SELECT 
          ue.id,
          ue.value,
          ue.type,
          ue.context,
          ue.confidence,
          ue.createdAt,
          ue.updatedAt
         FROM UserEntity ue
         WHERE ue.userId = ? 
           AND ue.agentId = ?
           AND ue.projectId IS NULL
         ORDER BY ue.type ASC, ue.updatedAt DESC
         LIMIT ? OFFSET ?`
      )
        .bind(userId, agentId, limit, offset)
        .all();

      const mapped: UserEntityWithId[] = entities.results.map((e: any) => ({
        id: e.id,
        type: e.type,
        value: e.value,
        context: e.context ?? undefined,
        confidence: e.confidence ?? undefined,
        createdAt: e.createdAt,
        updatedAt: e.updatedAt,
      }));

      return {
        data: mapped,
        pagination: {
          page,
          limit,
          total,
          totalPages,
          hasNextPage: page < totalPages,
          hasPrevPage: page > 1,
        },
      };
    } catch (error) {
      console.error("Error getting agent entities:", error);
      throw new Error("Failed to fetch agent entities");
    }
  }

  private async getEntityTypePriority(
    agentId: string,
    typeName: string,
    projectId?: number
  ): Promise<number> {
    try {
      // Hierarchical lookup: project-level first, then agent-level
      let entityType = null;

      if (projectId !== undefined) {
        // Check project-level first
        entityType = await this.env.DB.prepare(
          `SELECT priority FROM EntityType 
           WHERE agentId = ? AND projectId = ? AND typeName = ?`
        )
          .bind(agentId, projectId, typeName)
          .first();
      }

      if (!entityType) {
        // Check agent-level (fallback or primary check)
        entityType = await this.env.DB.prepare(
          `SELECT priority FROM EntityType 
           WHERE agentId = ? AND projectId IS NULL AND typeName = ?`
        )
          .bind(agentId, typeName)
          .first();
      }

      return (entityType?.priority as number) ?? ENTITY_PRIORITY.GENERAL;
    } catch (error) {
      console.error(
        `Error getting priority for entity type ${typeName}:`,
        error
      );
      return ENTITY_PRIORITY.GENERAL;
    }
  }

  // Step 2: Format entities for prompt (replicates userEntityService.formatEntitiesForPrompt)
  private async formatEntitiesForPrompt(
    entities: UserEntityWithId[],
    agentId: string,
    projectId?: number
  ): Promise<string> {
    try {
      if (!entities || entities.length === 0) {
        return "";
      }

      // Get unique entity types from the entities
      const entityTypes = [...new Set(entities.map((e) => e.type))];

      // Get entity type priorities
      let typePriorities: Record<string, number> = {};
      try {
        // Get priorities for all entity types
        await Promise.all(
          entityTypes.map(async (type) => {
            typePriorities[type] = await this.getEntityTypePriority(
              agentId,
              type,
              projectId
            );
          })
        );
      } catch (error) {
        console.error("Error getting entity type priorities:", error);
        // Fallback to default priorities
        typePriorities = {};
      }

      // Filter out excluded entities (priority -1) and group by type with priorities
      const entitiesWithPriority = entities
        .map((entity) => {
          const typeName = entity.type;
          return {
            ...entity,
            type: typeName,
            typePriority: typePriorities[typeName] ?? ENTITY_PRIORITY.GENERAL,
          };
        })
        .filter((entity) => entity.typePriority > ENTITY_PRIORITY.EXCLUDE); // Exclude priority -1

      if (entitiesWithPriority.length === 0) {
        return "";
      }

      // Group by type and sort by priority
      const entitiesByType = entitiesWithPriority.reduce<
        Record<string, { values: string[]; priority: number }>
      >((acc, entity) => {
        if (!acc[entity.type]) {
          acc[entity.type] = { values: [], priority: entity.typePriority };
        }
        acc[entity.type].values.push(entity.value);
        return acc;
      }, {});

      // Add priority-based formatting with section headers for high-priority types
      const result: string[] = [];
      const highPriorityTypes: string[] = [];
      const normalPriorityTypes: string[] = [];

      Object.entries(entitiesByType).forEach(([type, data]) => {
        const values = data.values.sort();
        const formattedLine = `**${type}:** ${values.join(", ")}`;

        if (data.priority >= ENTITY_PRIORITY.IMPORTANT) {
          // Priority 4 or higher
          highPriorityTypes.push(formattedLine);
        } else {
          normalPriorityTypes.push(formattedLine);
        }
      });

      // Sort by priority within each section
      highPriorityTypes.sort((a, b) => {
        const typeA = a.split(":")[0];
        const typeB = b.split(":")[0];
        const priorityA = entitiesByType[typeA]?.priority ?? 0;
        const priorityB = entitiesByType[typeB]?.priority ?? 0;
        return priorityB - priorityA;
      });

      normalPriorityTypes.sort((a, b) => {
        const typeA = a.split(":")[0];
        const typeB = b.split(":")[0];
        const priorityA = entitiesByType[typeA]?.priority ?? 0;
        const priorityB = entitiesByType[typeB]?.priority ?? 0;
        return priorityB - priorityA;
      });

      // Combine sections
      let formattedResult = "";
      if (highPriorityTypes.length > 0) {
        result.push("**Key Entities:**");
        result.push(...highPriorityTypes);

        if (normalPriorityTypes.length > 0) {
          result.push(""); // Empty line separator
          result.push("**Additional Context:**");
          result.push(...normalPriorityTypes);
        }
      } else {
        result.push(...normalPriorityTypes);
      }

      formattedResult = result.join("\n\n");

      if (formattedResult) {
        return `### USER ENTITIES & PREFERENCES:

${formattedResult}

**Instructions:**
- Adapt your communication style based on these personality traits and preferences
- Use learned information about user's interests and behavior patterns
- Customize responses to match user's preferred interaction style`;
      }

      return "";
    } catch (error) {
      console.error("Error formatting entities for prompt:", error);
      // Fallback to simple formatting
      const entitiesByType = entities.reduce<Record<string, string[]>>(
        (acc, entity) => {
          const typeName = entity.type;
          const value = entity.value;
          if (!acc[typeName]) {
            acc[typeName] = [];
          }
          acc[typeName].push(value);
          return acc;
        },
        {}
      );

      const simpleResult = Object.entries(entitiesByType)
        .map(([type, values]) => {
          return values.length > 3
            ? `**${type}:**\n  ${values.join("\n  ")}`
            : `**${type}:** ${values.join(", ")}`;
        })
        .join("\n\n");

      return simpleResult
        ? `### USER ENTITIES & PREFERENCES:

${simpleResult}

**Instructions:**
- Adapt your communication style based on these personality traits and preferences
- Use learned information about user's interests and behavior patterns`
        : "";
    }
  }

  // async onStart(): Promise<void> {
  //   try {
  //     console.log("🚀 Starting LumiLink MCP warmup...");

  //     // Warmup vectorize database with dummy queries
  //     await this.warmupVectorizeDB();

  //     console.log("✅ LumiLink MCP warmup completed");
  //   } catch (error) {
  //     console.error("⚠️ Warmup failed but continuing:", error);
  //   }
  // }

  private async warmupVectorizeDB(): Promise<void> {
    try {
      // Generate a dummy embedding for warmup
      const dummyText = "warmup query";
      const response = await this.env.AI.run("@cf/baai/bge-base-en-v1.5", {
        text: dummyText,
      });
      const dummyEmbedding = response.data[0];

      if (!dummyEmbedding) {
        console.log("❌ Could not generate dummy embedding for warmup");
        return;
      }

      // List of common namespaces to warmup
      const namespacesToWarmup = [
        "knowledge_base", // Knowledge base queries
        "memories-1", // Default user from fakePayload
        // Add other common namespaces as needed
      ];

      // Warmup each namespace with dummy query
      const warmupPromises = namespacesToWarmup.map(async (namespace) => {
        try {
          const startTime = Date.now();
          await this.env.VECTORIZE.query(dummyEmbedding, {
            namespace: namespace,
            topK: 1, // Minimal result set
            returnMetadata: false, // Skip metadata for speed
            filter: {}, // Empty filter
          });
          const endTime = Date.now();
          console.log(
            `🔥 Warmed up namespace "${namespace}" in ${endTime - startTime}ms`
          );
        } catch (err) {
          // Namespace might not exist, that's okay
          console.log(
            `⚡ Namespace "${namespace}" not found or empty (normal)`
          );
        }
      });

      // Execute all warmup queries in parallel
      await Promise.all(warmupPromises);

      console.log("🌟 Vectorize DB warmup completed");
    } catch (error) {
      console.error("🔥 Vectorize warmup failed:", error);
    }
  }
}

app.mount("/", async (req, env, ctx) => {
  const authHeader = req.headers.get("authorization");
  console.log(authHeader);
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
