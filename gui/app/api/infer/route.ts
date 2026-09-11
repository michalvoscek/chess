import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

export const runtime = "nodejs";

const INFER_TIMEOUT_MS = 120_000;

function findRepoRoot(start: string): string {
  let dir = start;
  for (let i = 0; i < 10; i++) {
    if (existsSync(path.join(dir, "infer.py"))) return dir;
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return start;
}

const REPO_ROOT = findRepoRoot(process.cwd());
const PYTHON_BIN = process.env.PYTHON_BIN ?? path.join(REPO_ROOT, "venv", "bin", "python");

interface InferBody {
  elo?: unknown;
  pgn?: unknown;
}

export async function POST(request: Request) {
  let body: InferBody;
  try {
    body = (await request.json()) as InferBody;
  } catch {
    return Response.json({ error: "invalid JSON body" }, { status: 400 });
  }

  const { elo, pgn } = body;
  if (typeof elo !== "number" || !Number.isInteger(elo)) {
    return Response.json({ error: "elo must be an integer" }, { status: 400 });
  }
  const pgnText = typeof pgn === "string" ? pgn : "";

  const result = await runInfer(elo, pgnText);
  if (result.error) {
    return Response.json({ error: result.error }, { status: 500 });
  }
  try {
    const moves = JSON.parse(result.stdout) as unknown;
    if (!Array.isArray(moves)) throw new Error("not an array");
    return Response.json({ moves });
  } catch {
    return Response.json({ error: "unexpected output from infer.py" }, { status: 500 });
  }
}

function runInfer(elo: number, pgn: string): Promise<{ stdout: string; error?: string }> {
  return new Promise((resolve) => {
    const proc = spawn(
      /*turbopackIgnore: true*/
      PYTHON_BIN,
      ["infer.py", "--elo", String(elo), "--pgn", pgn],
      {
        cwd: REPO_ROOT,
      },
    );
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => proc.kill("SIGKILL"), INFER_TIMEOUT_MS);
    proc.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    proc.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    proc.on("error", (err) => {
      clearTimeout(timer);
      resolve({ stdout, error: `failed to spawn infer.py: ${err.message}` });
    });
    proc.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) {
        resolve({ stdout });
        return;
      }
      const errorLine = stderr
        .trim()
        .split("\n")
        .filter((line) => line.startsWith("error:"))
        .at(-1);
      resolve({ stdout, error: errorLine ?? `infer.py exited with code ${code}` });
    });
  });
}
