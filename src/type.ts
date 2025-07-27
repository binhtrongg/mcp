export const ENTITY_PRIORITY = {
  EXCLUDE: -1,
  BACKGROUND: 0,
  MINOR: 1,
  CASUAL: 2,
  GENERAL: 3,
  IMPORTANT: 4,
  CRITICAL: 5,
} as const;

export interface UserEntity {
  type: string;
  value: string;
  context?: string;
  confidence?: number;
}

export interface UserEntityWithId extends UserEntity {
  id: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface PaginationParams {
  page?: number;
  limit?: number;
}

export interface PaginatedResult<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNextPage: boolean;
    hasPrevPage: boolean;
  };
}
