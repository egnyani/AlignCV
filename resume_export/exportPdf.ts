import { existsSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";

import chromium from "@sparticuz/chromium";
import puppeteer from "puppeteer-core";

import { renderResumeHtml } from "./renderResumeHtml";
import { PDF_CONFIG } from "./resumeLayout";
import type { ResumeData } from "./types";

function parseArgs(argv: string[]) {
  const args = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) continue;
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`Missing value for ${token}`);
    }
    args.set(token, value);
    index += 1;
  }

  const input = args.get("--input");
  const output = args.get("--output");
  if (!input || !output) {
    throw new Error(
      "Usage: npm run generate:resume-pdf -- --input <resume.json> --output <resume.pdf>",
    );
  }
  return { input, output };
}

async function launchBrowser() {
  const chromiumForPuppeteer = Object.assign(chromium, {
    defaultViewport: null,
    headless: "shell" as const,
  }) as typeof chromium & {
    defaultViewport: null;
    headless: "shell";
  };

  try {
    return await puppeteer.launch({
      args: chromiumForPuppeteer.args,
      defaultViewport: chromiumForPuppeteer.defaultViewport,
      executablePath: await chromiumForPuppeteer.executablePath(),
      headless: chromiumForPuppeteer.headless,
    });
  } catch (error) {
    if (
      process.platform !== "darwin" ||
      !(error instanceof Error) ||
      !error.message.includes("ENOEXEC")
    ) {
      throw error;
    }

    const localChromePath = [
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
      "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ].find((candidate) => existsSync(candidate));

    if (!localChromePath) {
      throw error;
    }

    return puppeteer.launch({
      executablePath: localChromePath,
      headless: true,
      args: ["--no-sandbox"],
    });
  }
}

function loadResumeData(inputPath: string): ResumeData {
  const raw = readFileSync(inputPath, "utf-8");
  return JSON.parse(raw) as ResumeData;
}

async function main() {
  const { input, output } = parseArgs(process.argv.slice(2));
  const inputPath = path.resolve(process.cwd(), input);
  const outputPath = path.resolve(process.cwd(), output);
  const outputDir = path.dirname(outputPath);
  mkdirSync(outputDir, { recursive: true });

  const data = loadResumeData(inputPath);
  const html = renderResumeHtml(data);
  const browser = await launchBrowser();

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 816, height: 600 });
    await page.setContent(html, { waitUntil: "networkidle0" });

    const pdfHeight = await page.evaluate(() => {
      const container = document.body.firstElementChild as HTMLElement | null;
      if (!container) return document.body.scrollHeight;
      return Math.ceil(container.getBoundingClientRect().height);
    });

    await page.pdf({
      path: outputPath,
      width: PDF_CONFIG.width,
      height: `${pdfHeight}px`,
      printBackground: PDF_CONFIG.printBackground,
      margin: PDF_CONFIG.margin,
    });
  } finally {
    await browser.close();
  }

  process.stdout.write(`${output}\n`);
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`${message}\n`);
  process.exitCode = 1;
});
