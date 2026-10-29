import "reflect-metadata";
import { Logger, ValidationPipe } from "@nestjs/common";
import { NestFactory } from "@nestjs/core";
import { AppModule } from "./app.module.js";

const PORT = Number(process.env["PORT"] ?? 3001);
/** Comma-separated origins; defaults to the Next.js dev server. */
const ORIGINS = (process.env["CORS_ORIGINS"] ?? "http://localhost:3000")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

async function bootstrap(): Promise<void> {
  const app = await NestFactory.create(AppModule);

  app.useGlobalPipes(
    new ValidationPipe({
      // Strip unknown keys and reject them loudly, so a typo'd field fails
      // visibly instead of being silently ignored.
      whitelist: true,
      forbidNonWhitelisted: true,
      transform: true,
      transformOptions: { enableImplicitConversion: true },
    }),
  );

  app.enableCors({ origin: ORIGINS });

  await app.listen(PORT);
  new Logger("bootstrap").log(`listening on http://localhost:${PORT}`);
}

await bootstrap();
