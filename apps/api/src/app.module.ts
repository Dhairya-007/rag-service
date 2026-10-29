import { Module } from "@nestjs/common";
import { AnswerService } from "./answer.service.js";
import { CorpusService } from "./corpus.service.js";
import { RagController } from "./rag.controller.js";

@Module({
  controllers: [RagController],
  providers: [CorpusService, AnswerService],
})
export class AppModule {}
