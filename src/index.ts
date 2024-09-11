import express, { Express, Request, Response } from "express";
import dotenv from "dotenv";
import multer from "multer";
import { createClient } from "@supabase/supabase-js";
// import { Database } from "../database.types";
import pdf from "pdf-parse";
import mammoth from "mammoth";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import path from "path";
import {OllamaEmbeddings} from '@langchain/ollama'

dotenv.config();

const app: Express = express();
app.use(express.json());
const port = process.env.PORT || 3000;
const upload = multer({ dest: "uploads/" });
const supabase = createClient(
  process.env.SUPABASE_URL || "",
  process.env.SUPABASE_KEY || ""
);

const openAI = new OpenAI({
  apiKey: process.env.OAI_KEY,
  temperature: 0.4,
  verbose: true,
  cache: false,
  maxTokens: -1,
  modelName: 'gpt-4o'
});

app.post("/create_container2", upload.array("files"), async (req, res) => {
  const files: any = req.files;
  if (!files || files.length === 0) {
    const container = await supabase.from("containers").insert({});
    return res
      .json({
        msg: "Container created",
        data: container.data,
      })
      .status(200);
  } else {
    try {
      const textPromises = files.map(async (file: any) => {
        const filePath = file.path;
        const fileExtension = file.originalname.split(".").pop().toLowerCase();
        let text = "";

        if (fileExtension === "pdf") {
          const pdfBuffer = require("fs").readFileSync(filePath);
          const pdfData = await pdf(pdfBuffer, { pagerender: render_page });
          text = pdfData.text;
        } else if (fileExtension === "docx") {
          const docBuffer = require("fs").readFileSync(filePath);
          const docData = await mammoth.extractRawText({ buffer: docBuffer });
          text = docData.value;
        }
        text = text
          .replace(/\n+/g, " ") // Replace multiple newlines with a single space
          .replace(/\s{2,}/g, " ") // Replace multiple spaces with a single space
          .trim(); // Trim leading and trailing spaces

        return text;
      });

      const texts: string[] = await Promise.all(textPromises);
      const singleString: string = texts.join("");

      function splitIntoChunks(content: string, chunkSize: number): string[] {
        const chunks = [];
        let currentIndex = 0;
        while (currentIndex < content.length) {
          chunks.push(content.slice(currentIndex, currentIndex + chunkSize));
          currentIndex += chunkSize;
        }
        return chunks;
      }

      const chunkSize = 14000;
      const contentChunks = splitIntoChunks(singleString, chunkSize);

      const proposition_template = `Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.
1. Split compound sentences into simple sentences. Maintain the original phrasing from the input whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifiers to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.
4. Present the results as a list of strings, formatted in JSON.
5. Retain as much information as possible.
Content: {Content}
`;

      const propositionPrompt =
        PromptTemplate.fromTemplate(proposition_template);

      const outputs: string[] = [];
      for (const chunk of contentChunks) {
        const answerChain = propositionPrompt
          .pipe(openAI)
          .pipe(new StringOutputParser());
        const output = await answerChain.invoke({ Content: chunk });
        outputs.push(output);
        break;
      }

      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 2000,
        chunkOverlap: 500,
      });

      const output = await splitter.createDocuments(outputs);
      const container = await supabase
        .from("containers")
        .insert({})
        .select("*");
      if (container.error) {
        console.log(container);
        return res.status(500);
      }
      const rpc = await supabase.rpc("vector-db-init", {
        table_id: container.data[0]["id"].toString(),
      });

      const vec = await SupabaseVectorStore.fromDocuments(
        output,
        new OpenAIEmbeddings({ apiKey: process.env.OAI_KEY }),
        {
          client: supabase,
          tableName: "table_" + container.data[0]["id"].toString(),
          queryName: "match_docs_" + container.data[0]["id"].toString(),
        }
      );
      const upd = await supabase
        .from("containers")
        .update({
          embeddings_table: container.data[0]["id"].toString(),
        })
        .eq("id", container.data[0]["id"]);

      return res
        .json({
          msg: "Container created with text chunks",
          data: container.data,
        })
        .status(200);
    } catch (error) {
      console.error(error);
      return res.status(500).json({ error: "Failed to process files" });
    }
  }
});

app.post("/create_container", upload.array("files"), async (req, res) => {
  const files: any = req.files;
  if (!files || files.length === 0) {
    const container = await supabase.from("containers").insert({});
    return res
      .json({
        msg: "Container created",
        data: container.data,
      })
      .status(200);
  } else {
    try {
      const textPromises = files.map(async (file: any) => {
        const filePath = file.path;
        const fileExtension = file.originalname.split(".").pop().toLowerCase();
        let text = "";

        if (fileExtension === "pdf") {
          const pdfBuffer = require("fs").readFileSync(filePath);
          const pdfData = await pdf(pdfBuffer, { pagerender: render_page });
          text = pdfData.text;
        } else if (fileExtension === "docx") {
          const docBuffer = require("fs").readFileSync(filePath);
          const docData = await mammoth.extractRawText({ buffer: docBuffer });
          text = docData.value;
        }
        text = text
          .replace(/\n+/g, " ") // Replace multiple newlines with a single space
          .replace(/\s{2,}/g, " ") // Replace multiple spaces with a single space
          .trim(); // Trim leading and trailing spaces

        return text;
      });

      const texts: string[] = await Promise.all(textPromises);

      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 500,
      });


      const output = await splitter.createDocuments(texts);

      const container = await supabase
        .from("containers")
        .insert({})
        .select("*");
      if (container.error) {
        console.log(container);
        return res.status(500);
      }

      const rpc = await supabase.rpc("vector-db-init", {
        table_id: container.data[0]["id"].toString(),
      });

      const vec = await SupabaseVectorStore.fromDocuments(
        output,
        new OpenAIEmbeddings({ apiKey: process.env.OAI_KEY }),
        {
          client: supabase,
          tableName: "table_" + container.data[0]["id"].toString(),
          queryName: "match_docs_" + container.data[0]["id"].toString(),
        }
      );
      const upd = await supabase
        .from("containers")
        .update({
          embeddings_table: container.data[0]["id"].toString(),
        })
        .eq("id", container.data[0]["id"]);

      return res
        .json({
          msg: "Container created with text chunks",
          data: container.data,
        })
        .status(200);
    } catch (error) {
      console.error(error);
      return res.status(500).json({ error: "Failed to process files" });
    }
  }
});

app.post("/generate_article", async (req, res) => {
  const prompt = req.body.prompt;
  const container = req.body.container;

  const answer_template = `Generate a high quality article about a company based upon the information provided. Use an example article as the template for the new article. TThe article should be of an enterprise nature. Do not make up any information not present in the company information. Make sure no incomplete points are made. Refuse to write about companies that are not present in the company information. Make sure to include specific data and stats where possible.
    The article should have a title, an introduction that summarizes the articles content, there should be headings where possible and it should have a conluding paragraph.
    article template: {article_template}
    company information: {company_info}
    prompt: {prompt}
    generated article: `;

  const data = await supabase
    .from("containers")
    .select("*")
    .eq("id", container);

  if (data.error) {
    console.log(data);
    return res.status(500);
  }

  const vector_store = await SupabaseVectorStore.fromExistingIndex(
    new OpenAIEmbeddings({ apiKey: process.env.OAI_KEY }),
    {
      client: supabase,
      tableName: "table_" + data.data[0].id,
      queryName: "match_docs_" + data.data[0].id,
    }
  );
  if (vector_store == null) {
    console.log(vector_store);
    return res.status(500);
  }
  const retriever = vector_store.asRetriever({ searchType: "mmr" });
  const answerPrompt = PromptTemplate.fromTemplate(answer_template);
  const retrieverChain = RunnableSequence.from([retriever, combineDocuments]);
  const ret = await retrieverChain.invoke(prompt);
  const answerChain = answerPrompt.pipe(openAI).pipe(new StringOutputParser());
  const article_template = `
EXAMPLE 1: The second new tea drink stock is coming: Chabaidao launches IPO, net profit compound annual growth rate reaches S21.6% 
As one of the hottest areas in recent years, the new tea beverage industry has developed rapidly, and many nationally renowned tea beverage brands have been born. Among the many brands, Sichuan Baicha Baidao Industrial Co., Ltd. (hereinafter referred to as "Chabaidao" or the "Company") has achieved continuous breakthroughs in business scale by virtue of its strong supply chain system and efficient franchise model. according toAccording to Frost & Sullivan's report,2023 In terms of annual retail sales, Chabaidao ranks third in China’s freshly made tea shop market.
Chabaidao launched its IPO in Hong Kong, with the stock code HK. Public information shows that in this IPO, Chabaidao plans to sell shares globally.H shares, each lot [] shares, expected to4 It will be listed on the Stock Exchange on [ ], and will become the second new tea drink stock in the Hong Kong stock market. 
Supply chain capabilities ensure product quality and build a solid foundation for development
A wide variety of products with stable quality is an important secret for Cha Baidao to conquer consumers. With the help of industry-leading R&D capabilities, the company has formed a rich product matrix consisting of classic tea drinks, seasonal tea drinks and regional tea drinks, which can meet the diversified consumption needs of different types of consumers for products. 
In order to ensure the stability and safety of product quality, Chabaidao achieves this goal by providing unified core raw materials to franchise stores across the country. To this end, the company has built industry-leading supply chain capabilities to ensure that it provides consumers with consistently high-quality products and provides solid support for the company's sustainable development. 
In the warehousing and logistics link, Chabaidao has established a nationwide warehousing and logistics network through self-operation and third-party cooperation, which can achieve logistics storage and delivery capabilities for most stores across the country with a delivery frequency of twice a week or more. At the same time, the company also meets the temperature and humidity requirements of different raw materials through full-scale refined warehousing management.
Reduce storage needs, and use digital means to realize visualization and traceability of the entire logistics process, reduce distribution losses, achieve stable quality of raw materials, and improve operational efficiency and profitability. In the procurement process, the company has established the ability to self-pick and prepare fresh fruits, and continues to strengthen cooperation with leading suppliers on core raw materials to achieve quality and stable supply of raw materials. 
Efficient franchise model helps scale expansion and strong performance growth
In terms of market expansion, Chabaidao mainly expands the market through franchising. The company adheres to the business philosophy of mutual benefit and win-win, and provides support to franchisees in many aspects such as store location selection, decoration and opening, staff training, store operations, unified takeout business and in-store business empowerment, and marketing, empowering franchisees to grow healthily. . According to Frost & Sullivan's report,2021 Year,2022 year and2023 In 2018, the closure rate of Chabaidao franchise stores was only0.2%、1.1%and 2.3%, far below the industry average. 
Relying on the effective franchise model, Chabaidao's store network has increased from one to a hundred, and then to thousands, with increasing levels. The number of its stores is2021 at the beginning of the year2,242 Home expands to2023 end of year7,801 home, which has grown by more than 30% in three years.3 times and further expanded to7,927 Home. 
It is worth mentioning that Chabaidao has also achieved strong growth in performance while rapidly expanding its scale.2021 New Year's Eve2023 During the year, the company’s revenue consisted of RMB3,644.2 million increased to5,704.3 million yuan, with a compound annual growth rate of25.1% ; Net profit is represented by RMB778.5 million increased to1150.8 million yuan, with a compound annual growth rate of21.6%, profitability continues to grow, further proving the correctness of the company's business model. 
In the medium to long term, the market size of China's ready-made tea shop industry will still maintain double-digit growth. According to Frost & Sullivan’s report, the tea shop industry2024 New Year's Eve2028 The compound annual growth rate is expected to be as high as15.4%. Cha Baidao continues to capture consumer groups through high-quality and rich products, and has built on its efficient supply chain system and processing
With the support of the alliance model, it is expected to usher in “snowball” growth opportunities.`;

  const resp = await answerChain.invoke({
    article_template,
    company_info: ret,
    prompt,
  });

  return res
    .json({
      msg: "Article generated",
      data: {
        article: resp,
        rating: await rateArticle(resp)
      },
    })
    .status(200);
});

async function rateArticle(article: string) {
  const rate_template = `I have a program that generates Enterprise style articles about companies. They are usually about the companies finances, stock performance, franchisee conditions, profit, etc. I want you to rate the generated articles out of 10. Consider the generated content, the style, structure, tone and coherency of the articles. You have no other or previous context.  Article: {article}`
  const rate_prompt = PromptTemplate.fromTemplate(rate_template)
  const rateChain = rate_prompt.pipe(openAI).pipe(new StringOutputParser());
  return await rateChain.invoke({article})
}

app.post('/rate_article', async(req, res) => {
  return res.json({rating: await rateArticle(req.body.article)}).status(200)
})

app.use(express.static(path.join(__dirname, 'public')));

// Default route to serve index.html
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(port, () => {
  console.log(`[server]: Server is running at http://localhost:${port}`);
});

function render_page(pageData: any) {
  let render_options = {
    //replaces all occurrences of whitespace with standard spaces (0x20). The default value is `false`.
    normalizeWhitespace: true,
    //do not attempt to combine same line TextItem's. The default value is `false`.
    disableCombineTextItems: false,
  };

  return pageData
    .getTextContent(render_options)
    .then(function (textContent: any) {
      let lastY,
        text = "";
      for (let item of textContent.items) {
        if (lastY == item.transform[5] || !lastY) {
          text += item.str + " ";
        } else {
          text += "\n" + item.str + " ";
        }
        lastY = item.transform[5];
      }
      return text;
    });
}

function combineDocuments(docs: any[]) {
  return docs.map((doc) => doc["pageContent"]).join("\n\n");
}
