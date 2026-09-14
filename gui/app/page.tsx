"use client";

import { useEffect, useRef, useState } from "react";
import { Chess, type Move, type Square } from "chess.js";
import type { Api } from "@lichess-org/chessground/api";
import type { Dests, Key } from "@lichess-org/chessground/types";
import "@lichess-org/chessground/assets/chessground.base.css";
import "@lichess-org/chessground/assets/chessground.brown.css";
import "@lichess-org/chessground/assets/chessground.cburnett.css";

interface InferMove {
  move: string;
  p: number;
}

interface PendingPromotion {
  from: Key;
  to: Key;
}

const PROMOTION_PIECES = ["q", "r", "b", "n"] as const;
const PROMOTION_GLYPHS: Record<(typeof PROMOTION_PIECES)[number], string> = {
  q: "♛",
  r: "♜",
  b: "♝",
  n: "♞",
};

function toDests(chess: Chess): Dests {
  const dests: Dests = new Map();
  for (const move of chess.moves({ verbose: true }) as Move[]) {
    const list = dests.get(move.from);
    if (list) list.push(move.to);
    else dests.set(move.from, [move.to]);
  }
  return dests;
}

function buildPgn(history: string[]): string {
  let pgn = "";
  for (let i = 0; i < history.length; i++) {
    if (i % 2 === 0) pgn += `${i / 2 + 1}. `;
    pgn += `${history[i]} `;
  }
  return pgn.trim();
}

function roundElo(input: string): number | null {
  const value = Number(input);
  if (!Number.isFinite(value)) return null;
  const clamped = Math.min(Math.max(value, 0), 4000);
  return 100 * Math.floor(clamped / 100 + 0.5);
}

function weightedSample(moves: InferMove[]): InferMove {
  const total = moves.reduce((sum, move) => sum + move.p, 0);
  let roll = Math.random() * total;
  for (const move of moves) {
    roll -= move.p;
    if (roll <= 0) return move;
  }
  return moves[moves.length - 1];
}

export default function Home() {
  const [game, setGame] = useState(() => new Chess());

  const boardRef = useRef<HTMLDivElement | null>(null);
  const apiRef = useRef<Api | null>(null);
  const applyStateRef = useRef<() => void>(() => {});
  const userMoveRef = useRef<(orig: Key, dest: Key) => void>(() => {});
  const playNextRef = useRef<() => void>(() => {});
  const attemptedFenRef = useRef<string | null>(null);

  const [fen, setFen] = useState(game.fen());
  const [eloInput, setEloInput] = useState("1500");
  const [autoplay, setAutoplay] = useState<"none" | "white" | "black">("none");
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [promotion, setPromotion] = useState<PendingPromotion | null>(null);

  const history = game.history();
  const gameOver = game.isGameOver();
  const dests = toDests(game);
  const appliedElo = roundElo(eloInput);

  function playUserMove(from: Key, to: Key, piece?: (typeof PROMOTION_PIECES)[number]) {
    try {
      game.move({ from, to, promotion: piece });
    } catch {
      setError("illegal move");
      return;
    }
    setError(null);
    attemptedFenRef.current = null;
    setFen(game.fen());
  }

  function handleUserMove(orig: Key, dest: Key) {
    const piece = game.get(orig as Square);
    if (
      piece?.type === "p" &&
      ((piece.color === "w" && dest.endsWith("8")) || (piece.color === "b" && dest.endsWith("1")))
    ) {
      apiRef.current?.set({ fen: game.fen() });
      setPromotion({ from: orig, to: dest });
      return;
    }
    playUserMove(orig, dest);
  }

  useEffect(() => {
    userMoveRef.current = handleUserMove;
    playNextRef.current = playNext;
    applyStateRef.current = () => {
      const last = game.history({ verbose: true }).at(-1);
      apiRef.current?.set({
        fen: game.fen(),
        turnColor: game.turn() === "w" ? "white" : "black",
        check: game.inCheck(),
        lastMove: last ? [last.from, last.to] : undefined,
        movable: {
          free: false,
          color:
            autoplay === "none"
              ? "both"
              : autoplay === "white"
                ? "black"
                : "white",
          showDests: true,
          dests: thinking || promotion || gameOver ? undefined : dests,
          events: {
            after: (orig, dest) => userMoveRef.current(orig, dest),
          },
        },
      });
    };
  });

  useEffect(() => {
    applyStateRef.current();
  }, [fen, thinking, promotion, gameOver, dests]);

  useEffect(() => {
    if (autoplay === "none" || thinking || gameOver) return;
    if (game.turn() !== (autoplay === "white" ? "w" : "b")) return;
    if (attemptedFenRef.current === game.fen()) return;
    attemptedFenRef.current = game.fen();
    playNextRef.current();
  }, [autoplay, game, fen, thinking, gameOver]);

  useEffect(() => {
    let disposed = false;
    import("@lichess-org/chessground").then(({ Chessground }) => {
      if (disposed || !boardRef.current) return;
      apiRef.current = Chessground(boardRef.current, { coordinates: true });
      applyStateRef.current();
    });
    return () => {
      disposed = true;
      apiRef.current?.destroy();
      apiRef.current = null;
    };
  }, []);

  async function playNext() {
    if (thinking || gameOver) return;
    const elo = roundElo(eloInput);
    if (elo === null) {
      setError("enter a valid elo number");
      return;
    }
    setThinking(true);
    setError(null);
    try {
      const res = await fetch("/api/infer", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ elo, pgn: buildPgn(history) }),
      });
      const data = (await res.json()) as { moves?: InferMove[]; error?: string };
      if (!res.ok || !data.moves) {
        throw new Error(data.error ?? "inference failed");
      }
      const picked = weightedSample(data.moves);
      const verbose = (game.moves({ verbose: true }) as Move[]).find(
        (move) => move.san === picked.move,
      );
      if (!verbose) {
        throw new Error(`model suggested illegal move ${picked.move}`);
      }
      game.move({ from: verbose.from, to: verbose.to, promotion: verbose.promotion });
      setFen(game.fen());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setThinking(false);
    }
  }

  function undo() {
    if (thinking || !history.length) return;
    game.undo();
    setPromotion(null);
    setError(null);
    attemptedFenRef.current = null;
    setFen(game.fen());
  }

  function restart() {
    if (thinking) return;
    const fresh = new Chess();
    setGame(fresh);
    setPromotion(null);
    setError(null);
    attemptedFenRef.current = null;
    setFen(fresh.fen());
  }

  const buttonClass =
    "cursor-pointer flex-auto rounded-lg border border-zinc-600 bg-zinc-800 px-4 py-2 text-sm font-medium text-zinc-100 transition-colors hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40";

  return (
    <main className="flex w-full flex-1 flex-col items-center justify-center gap-8 p-6 lg:flex-row lg:items-start lg:justify-center">
      <div className="w-full max-w-[480px] self-center">
        <div ref={boardRef} className="aspect-square w-full" />
      </div>
      <div className="flex w-full max-w-[480px] flex-col gap-4 lg:w-72 lg:max-w-none">
        <div className="flex flex-col gap-1">
          <label htmlFor="autoplay" className="text-sm font-medium text-zinc-400">
            autoplay
          </label>
          <select
            id="autoplay"
            value={autoplay}
            onChange={(event) => {
              setAutoplay(event.target.value as "none" | "white" | "black");
              attemptedFenRef.current = null;
            }}
            className="rounded-lg border border-zinc-600 bg-zinc-900 px-3 py-2 text-lg text-zinc-100 outline-none focus:border-zinc-400"
          >
            <option value="none">no autoplay</option>
            <option value="white">autoplay white</option>
            <option value="black">autoplay black</option>
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="elo" className="text-sm font-medium text-zinc-400">
            elo (side to move)
          </label>
          <input
            id="elo"
            type="number"
            min={0}
            max={4000}
            step={100}
            value={eloInput}
            onChange={(event) => setEloInput(event.target.value)}
            className="rounded-lg border border-zinc-600 bg-zinc-900 px-3 py-2 text-lg text-zinc-100 outline-none focus:border-zinc-400"
          />
          {appliedElo !== null && (
            <span className="text-xs text-zinc-500">applied as {appliedElo}</span>
          )}
        </div>
        <div className="flex flex-col gap-2">
          <button onClick={playNext} disabled={thinking || gameOver} className={buttonClass}>
            {thinking ? "thinking…" : "play next move"}
          </button>
          <div className="w-full flex flex-row gap-2">
            <button onClick={undo} disabled={thinking || !history.length} className={buttonClass}>
              undo
            </button>
            <button onClick={restart} disabled={thinking} className={buttonClass}>
              restart
            </button>
          </div>
        </div>
        {error && <p className="text-sm text-red-400">{error}</p>}
        <div className="h-96 rounded-lg border border-zinc-700  p-3 text-sm leading-7 text-zinc-900 overflow-auto">
          {history.length === 0 && <span className="text-zinc-500">no moves yet</span>}
          {Array.from({ length: Math.ceil(history.length / 2) }, (_, i) => (
            <div key={i} className="mr-3 whitespace-nowrap">
              <span className="text-zinc-500">{i + 1}.</span> {history[i * 2]}{" "}
              {history[i * 2 + 1] ?? ""}
            </div>
          ))}
        </div>
      </div>
      {promotion && (
        <div
          className="fixed inset-0 z-10 flex items-center justify-center bg-black/60"
          onClick={() => setPromotion(null)}
        >
          <div
            className="flex gap-2 rounded-xl bg-zinc-100 p-4 shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            {PROMOTION_PIECES.map((piece) => (
              <button
                key={piece}
                onClick={() => {
                  const pending = promotion;
                  setPromotion(null);
                  playUserMove(pending.from, pending.to, piece);
                }}
                className="flex h-16 w-16 cursor-pointer items-center justify-center rounded-lg text-5xl text-zinc-900 transition-colors hover:bg-zinc-200"
                aria-label={`promote to ${piece}`}
              >
                {PROMOTION_GLYPHS[piece]}
              </button>
            ))}
          </div>
        </div>
      )}
    </main>
  );
}
