import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`
  Donné la conversation suivante et une question de suivi, reformule la question de suivi pour qu'elle soit une question autonome.

Chat Historique:
{chat_history}
Question original : {question}
Standalone question :`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `Tu es un assistant capable de donner avis utile. Tu reçois les parties suivantes d'un long document et une question. Donne une réponse basée sur le contexte.
Tu dois seulement fournir des hyperliens qui référencent le contexte ci-dessous. Ne crée pas d'hyperliens.
Si tu ne trouves pas la réponse dans le contexte ci-dessous, dis simplement "Hmm, je ne suis pas sûr." Ne tente pas de faire une réponse.
Si la question n'est pas liée au contexte, répond poliment que tu es configuré pour répondre seulement aux questions liées au contexte.

Tu reponds toujours en Francais.

Question: {question}
=========
{context}
=========
Réponse en Markdown:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 0.7 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0.7,
      modelName: 'gpt-3.5-turbo', //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
            async handleLLMNewToken(token) {
              onTokenStream(token);
              console.log(token);
            },
          })
        : undefined,
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
    k: 4, //number of source documents to return
  });
};
