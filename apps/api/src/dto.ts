import {
  IsIn,
  IsInt,
  IsOptional,
  IsString,
  Matches,
  Max,
  MaxLength,
  Min,
  MinLength,
} from "class-validator";

export class IngestDto {
  /**
   * Doubles as the citation prefix (`handbook#3`), so it is restricted to
   * characters that survive a URL and a bracketed citation unambiguously.
   */
  @IsString()
  @Matches(/^[\w.-]{1,64}$/, {
    message: "id must be 1-64 characters of letters, digits, dot, dash or underscore",
  })
  id!: string;

  @IsString()
  @MinLength(1)
  @MaxLength(200)
  title!: string;

  @IsString()
  @MinLength(1)
  @MaxLength(1_000_000)
  text!: string;

  @IsOptional()
  @IsString()
  @MaxLength(500)
  source?: string;
}

export class SearchDto {
  @IsString()
  @MinLength(2)
  @MaxLength(1_000)
  query!: string;

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(50)
  limit?: number;

  @IsOptional()
  @IsIn(["hybrid", "lexical", "dense"])
  mode?: "hybrid" | "lexical" | "dense";
}

export class AskDto {
  @IsString()
  @MinLength(2)
  @MaxLength(2_000)
  question!: string;

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(20)
  limit?: number;
}
