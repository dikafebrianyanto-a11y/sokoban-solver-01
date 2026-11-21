// sokoban-solver: a push-optimal Sokoban solver for make.fun's sokoban-solver-01-run-20261121.
//
// Design (written from first principles for this competition; the house baseline is a plain
// move-based A* with no deadlock knowledge, so it is beaten on both counts below):
//
// 1. DECOMPOSITION. Every generated level is a hallway (one row, where the player starts) with
//    1-4 rooms off it, each inside a 6x6 slot, one door each, 3-4 boxes and as many goals. A box
//    pushed out of a room through its door can never come back, so in every optimal solution each
//    box stays in its room — the level's push-optimal count P* is the sum of its rooms' optima.
//    This solver therefore splits the level into independent rooms, solves each room
//    push-optimally, and stitches the rooms together with hallway walks (walks cost no pushes,
//    so only the per-room push counts matter for the score).
//
// 2. PUSH-BASED A*. Each room is solved by A* over pushes, not steps: a state is the sorted box
//    positions plus the player's reachable region (the player cell is normalized to the minimum
//    cell of its region, since any cell of the region reaches any other by walking). The
//    heuristic is the minimum assignment of boxes to goals by BFS distance over the room's floor
//    (each push moves one box one cell, so the assignment sum can drop by at most one per push:
//    admissible, hence the first goal state popped is push-optimal). Dead states are pruned:
//    boxes on squares from which no goal is reachable, corner deadlocks, 2x2 wall/box blocks,
//    and boxes frozen against a wall whose whole wall segment holds no goal.
//    The search never reads the clock: budgets are node counts, so two runs are byte-identical.
//
// 3. EXIT. After a room is solved the player must walk back out through its door to reach the
//    next room. The search first looks for the unconstrained optimum and takes it when the
//    player can still reach the door afterwards; otherwise it re-searches with the exit required.
//    The last room needs no exit. If a room cannot be solved, the level is answered empty
//    (an unsolved replay scores 0 either way, and an empty answer is always valid).
//
// Entry contract: reads /input/<id>/level.txt (or MAKEFUN_INPUT_DIR) and writes
// /output/<id>/solution.txt (or MAKEFUN_OUTPUT_DIR), one lowercase LURD move per step.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Node budgets are counts, never time, so the output is deterministic.
const EXPANSIONS_PER_ATTEMPT = 400_000;
const EXPANSIONS_GLOBAL = 2_000_000;

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const CH = ['u', 'd', 'l', 'r'];

interface Board {
  w: number;
  h: number;
  wall: Uint8Array;
  goal: Uint8Array;
  boxes: number[];
  player: number;
}

/** Parse XSB exactly the way the checker does (short rows are walls past their end). */
function parseLevel(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const wall = new Uint8Array(w * h).fill(1);
  const goal = new Uint8Array(w * h);
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

interface Room {
  cells: number[]; // floor cells of the room, sorted
  mask: Uint8Array; // 1 on room cells
  door: number; // room cell adjacent to the hallway
  entry: number; // hallway cell adjacent to the door
  boxes: number[];
  goals: number[];
}

/**
 * Split a level into rooms. The hallway is the player's row; each room is a connected component
 * of the floor off that row. A room's door is its cell touching a hallway floor cell.
 * Returns null when the level does not have the generated shape.
 */
function decompose(board: Board): Room[] | null {
  const { w, h, wall, goal, boxes, player } = board;
  if (player < 0) return null;
  const hallRow = Math.floor(player / w);
  const seen = new Uint8Array(w * h);
  const rooms: Room[] = [];
  const roomOfBox = new Map<number, number>();
  for (let start = 0; start < w * h; start++) {
    if (wall[start] || Math.floor(start / w) === hallRow || seen[start]) continue;
    // Flood-fill one connected component off the hallway row.
    const cells: number[] = [];
    const stack = [start];
    seen[start] = 1;
    while (stack.length > 0) {
      const c = stack.pop()!;
      cells.push(c);
      const x = c % w;
      const y = Math.floor(c / w);
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d];
        const ny = y + DY[d];
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const n = ny * w + nx;
        if (wall[n] || Math.floor(n / w) === hallRow || seen[n]) continue;
        seen[n] = 1;
        stack.push(n);
      }
    }
    cells.sort((a, b) => a - b);
    // The door is the room cell touching a hallway floor cell (smallest such cell wins).
    let door = -1;
    let entry = -1;
    for (const c of cells) {
      const x = c % w;
      const y = Math.floor(c / w);
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d];
        const ny = y + DY[d];
        if (ny !== hallRow || nx < 0 || nx >= w) continue;
        const n = ny * w + nx;
        if (!wall[n]) {
          door = c;
          entry = n;
          break;
        }
      }
      if (door >= 0) break;
    }
    if (door < 0) return null;
    const mask = new Uint8Array(w * h);
    for (const c of cells) mask[c] = 1;
    const roomBoxes = boxes.filter((b) => mask[b] === 1);
    const roomGoals = cells.filter((c) => goal[c] === 1);
    if (roomBoxes.length === 0 || roomGoals.length === 0) return null;
    const idx = rooms.length;
    for (const b of roomBoxes) roomOfBox.set(b, idx);
    rooms.push({ cells, mask, door, entry, boxes: roomBoxes, goals: roomGoals });
  }
  if (rooms.length === 0) return null;
  // Every box must belong to exactly one room.
  if (roomOfBox.size !== boxes.length) return null;
  // Rooms in door-column order (left to right), then door row: the visit order.
  rooms.sort((a, b) => (a.door % w) - (b.door % w) || Math.floor(a.door / w) - Math.floor(b.door / w));
  return rooms;
}

/** All permutations of [0..n-1], generated once per room (n <= 4). */
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const a = Array.from({ length: n }, (_, i) => i);
  const gen = (k: number): void => {
    if (k === n) {
      out.push(a.slice());
      return;
    }
    for (let i = k; i < n; i++) {
      [a[k], a[i]] = [a[i], a[k]];
      gen(k + 1);
      [a[k], a[i]] = [a[i], a[k]];
    }
  };
  gen(0);
  return out;
}

/**
 * One room's search context: the cells the player may stand on (room cells + entry cell),
 * BFS distances from each goal over those cells, and the deadlock tables.
 */
class RoomSearch {
  readonly board: Board;
  readonly room: Room;
  readonly w: number;
  readonly allowed: Uint8Array; // room cells + entry cell
  readonly goals: number[];
  readonly dist: Int32Array[]; // dist[g][cell], -1 when unreachable
  readonly perms: number[][];
  readonly goalMask: Uint8Array;

  constructor(board: Board, room: Room) {
    this.board = board;
    this.room = room;
    const { w, h } = board;
    this.w = w;
    this.allowed = new Uint8Array(w * h);
    for (const c of room.cells) this.allowed[c] = 1;
    this.allowed[room.entry] = 1;
    this.goals = room.goals;
    this.goalMask = new Uint8Array(w * h);
    for (const g of this.goals) this.goalMask[g] = 1;
    // BFS distances from each goal over the allowed cells (boxes ignored).
    this.dist = this.goals.map((g) => {
      const d = new Int32Array(w * h).fill(-1);
      const q = [g];
      d[g] = 0;
      for (let qi = 0; qi < q.length; qi++) {
        const c = q[qi]!;
        const x = c % w;
        const y = Math.floor(c / w);
        for (let dir = 0; dir < 4; dir++) {
          const nx = x + DX[dir];
          const ny = y + DY[dir];
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          const n = ny * w + nx;
          if (this.allowed[n] === 0 || d[n] >= 0) continue;
          d[n] = d[c]! + 1;
          q.push(n);
        }
      }
      return d;
    });
    this.perms = permutations(this.goals.length);
  }

  /** Admissible heuristic: min assignment of boxes to goals by BFS distance. Infinity = dead. */
  heuristic(boxes: number[]): number {
    if (boxes.length !== this.goals.length) {
      // Defensive: counts differ (never for generated rooms). Sum of nearest-goal distances.
      let s = 0;
      for (const b of boxes) {
        let m = Infinity;
        for (const d of this.dist) if (d[b]! >= 0 && d[b]! < m) m = d[b]!;
        if (m === Infinity) return Infinity;
        s += m;
      }
      return s;
    }
    let best = Infinity;
    for (const p of this.perms) {
      let s = 0;
      for (let i = 0; i < boxes.length; i++) {
        const d = this.dist[p[i]!]![boxes[i]!]!;
        if (d < 0) {
          s = Infinity;
          break;
        }
        s += d;
      }
      if (s < best) best = s;
    }
    return best;
  }

  /** True when a box sits on a square no push can ever save (not a goal). */
  private deadBox(b: number, boxSet: Set<number>): boolean {
    if (this.goalMask[b] === 1) return false;
    const { w, h } = this;
    const x = b % w;
    const y = Math.floor(b / w);
    const wallAt = (nx: number, ny: number): boolean => {
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) return true;
      return this.allowed[ny * w + nx] === 0;
    };
    // Corner: walls on two perpendicular sides.
    if ((wallAt(x, y - 1) && wallAt(x - 1, y)) || (wallAt(x, y - 1) && wallAt(x + 1, y)) ||
        (wallAt(x, y + 1) && wallAt(x - 1, y)) || (wallAt(x, y + 1) && wallAt(x + 1, y))) return true;
    // 2x2 block of walls/boxes holding a box off its goal: no box in it can ever move.
    for (const [ox, oy] of [[-1, -1], [0, -1], [-1, 0], [0, 0]] as const) {
      const cells = [
        (y + oy) * w + (x + ox),
        (y + oy) * w + (x + ox + 1),
        (y + oy + 1) * w + (x + ox),
        (y + oy + 1) * w + (x + ox + 1),
      ];
      let allBlocked = true;
      let anyBoxOffGoal = false;
      for (const c of cells) {
        const cx = c % w;
        const cy = Math.floor(c / w);
        if (cx < 0 || cx >= w || cy < 0 || cy >= h || this.allowed[c] === 0) continue; // a wall cell
        if (!boxSet.has(c)) {
          allBlocked = false;
          break;
        }
        if (this.goalMask[c] === 0) anyBoxOffGoal = true;
      }
      if (allBlocked && anyBoxOffGoal) return true;
    }
    // Frozen against a wall: walls on both sides of one axis, and no goal anywhere on the
    // wall segment the box can slide along.
    if (wallAt(x, y - 1) && wallAt(x, y + 1)) {
      let hasGoal = false;
      for (let nx = x; nx >= 0 && this.allowed[y * w + nx] === 1; nx--) if (this.goalMask[y * w + nx] === 1) hasGoal = true;
      for (let nx = x + 1; nx < w && this.allowed[y * w + nx] === 1; nx++) if (this.goalMask[y * w + nx] === 1) hasGoal = true;
      if (!hasGoal) return true;
    }
    if (wallAt(x - 1, y) && wallAt(x + 1, y)) {
      let hasGoal = false;
      for (let ny = y; ny >= 0 && this.allowed[ny * w + x] === 1; ny--) if (this.goalMask[ny * w + x] === 1) hasGoal = true;
      for (let ny = y + 1; ny < this.board.h && this.allowed[ny * w + x] === 1; ny++) if (this.goalMask[ny * w + x] === 1) hasGoal = true;
      if (!hasGoal) return true;
    }
    return false;
  }

  /** True when any box in the state is dead (or unreachable from every goal). */
  private deadState(boxes: number[]): boolean {
    const boxSet = new Set(boxes);
    for (const b of boxes) {
      if (this.goalMask[b] === 1) continue;
      let reachable = false;
      for (const d of this.dist) if (d[b]! >= 0) { reachable = true; break; }
      if (!reachable) return true;
      if (this.deadBox(b, boxSet)) return true;
    }
    return false;
  }

  /**
   * The player's reachable region from `from`, over allowed cells avoiding `boxes`.
   * Returns the region's minimum cell (the state's canonical representative) and the
   * parent map for walk reconstruction (parent[cell] = previous cell on the BFS tree).
   */
  private region(from: number, boxes: number[]): { rep: number; parent: Int32Array } {
    const { w, h } = this.board;
    const boxSet = new Set(boxes);
    const parent = new Int32Array(w * h).fill(-2);
    const q = [from];
    parent[from] = -1;
    let rep = from;
    for (let qi = 0; qi < q.length; qi++) {
      const c = q[qi]!;
      if (c < rep) rep = c;
      const x = c % w;
      const y = Math.floor(c / w);
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d];
        const ny = y + DY[d];
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const n = ny * w + nx;
        if (this.allowed[n] === 0 || boxSet.has(n) || parent[n] !== -2) continue;
        parent[n] = c;
        q.push(n);
      }
    }
    return { rep, parent };
  }

  /** Walk moves from `from` to `to` through the region's BFS tree (both cells in the region). */
  private walk(from: number, to: number, parent: Int32Array): string {
    const { w } = this;
    const path: number[] = [];
    for (let c = to; c !== from && c >= 0; c = parent[c]!) path.push(c);
    path.reverse();
    let moves = '';
    let cur = from;
    for (const n of path) {
      const dx = (n % w) - (cur % w);
      const dy = Math.floor(n / w) - Math.floor(cur / w);
      moves += dx === 1 ? 'r' : dx === -1 ? 'l' : dy === 1 ? 'd' : 'u';
      cur = n;
    }
    return moves;
  }

  /**
   * Push-optimal A* for one room. `needsExit` requires the final player position to reach the
   * door (so the player can leave for the next room). `weight` > 1 trades optimality for speed;
   * `greedy` orders purely by heuristic. `budget` caps expansions (count-based, deterministic).
   * Always reports the expansions used, even when no solution was found.
   */
  search(
    startBoxes: number[],
    startPlayer: number,
    needsExit: boolean,
    weight: number,
    greedy: boolean,
    budget: number,
  ): { moves: string | null; player: number; expansions: number } {
    const { w } = this;
    const startH = this.heuristic(startBoxes);
    if (startH === Infinity) return { moves: null, player: startPlayer, expansions: 0 };
    // Node storage (parallel arrays): boxes, player, g, parent, moves from parent.
    const nBoxes: number[][] = [startBoxes.slice()];
    const nPlayer: number[] = [startPlayer];
    const nG: number[] = [0];
    const nParent: number[] = [-1];
    const nMoves: string[] = [''];
    const best = new Map<string, number>();
    const startRegion = this.region(startPlayer, startBoxes);
    best.set(startBoxes.join(',') + '|' + startRegion.rep, 0);
    // Binary heap of [priority, h, seq, node]. Lower priority first, then lower h, then older.
    const heap: [number, number, number, number][] = [];
    let seq = 0;
    const heapLess = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
      a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
    const heapPush = (e: [number, number, number, number]): void => {
      let i = heap.length;
      heap.push(e);
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (!heapLess(heap[i]!, heap[p]!)) break;
        [heap[i], heap[p]] = [heap[p]!, heap[i]!];
        i = p;
      }
    };
    const heapPop = (): [number, number, number, number] => {
      const top = heap[0]!;
      const last = heap.pop()!;
      if (heap.length > 0) {
        heap[0] = last;
        for (let i = 0; ; ) {
          let c = 2 * i + 1;
          if (c >= heap.length) break;
          if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
          if (!heapLess(heap[c]!, heap[i]!)) break;
          [heap[i], heap[c]] = [heap[c]!, heap[i]!];
          i = c;
        }
      }
      return top;
    };
    const prio = (g: number, h: number): number => (greedy ? h : g + weight * h);
    heapPush([prio(0, startH), startH, seq++, 0]);
    let expansions = 0;
    while (heap.length > 0 && expansions < budget) {
      const [, , , s] = heapPop();
      const boxes = nBoxes[s]!;
      const player = nPlayer[s]!;
      const g = nG[s]!;
      const reg = this.region(player, boxes);
      const key = boxes.join(',') + '|' + reg.rep;
      if (best.get(key) !== g) continue; // stale: a better route to this state was recorded
      if (boxes.every((b) => this.goalMask[b] === 1)) {
        if (!needsExit || reg.parent[this.room.door] !== -2) {
          let moves = '';
          for (let t = s; t > 0; t = nParent[t]!) moves = nMoves[t]! + moves;
          return { moves, player, expansions };
        }
      }
      expansions++;
      for (let bi = 0; bi < boxes.length; bi++) {
        const b = boxes[bi]!;
        const bx = b % w;
        const by = Math.floor(b / w);
        for (let d = 0; d < 4; d++) {
          const tx = bx + DX[d];
          const ty = by + DY[d];
          const target = ty * w + tx;
          // A push must land inside the room (never out through the door) on a free cell.
          if (this.room.mask[target] !== 1 || boxes.includes(target)) continue;
          const sx = bx - DX[d];
          const sy = by - DY[d];
          const stance = sy * w + sx;
          if (reg.parent[stance] === -2) continue; // player cannot get behind the box
          const nb = boxes.slice();
          nb[bi] = target;
          nb.sort((x, y) => x - y);
          if (this.deadState(nb)) continue;
          const h = this.heuristic(nb);
          if (h === Infinity) continue;
          const nReg = this.region(b, nb);
          const nKey = nb.join(',') + '|' + nReg.rep;
          const seen = best.get(nKey);
          if (seen !== undefined && seen <= g + 1) continue;
          best.set(nKey, g + 1);
          const t = nBoxes.length;
          nBoxes.push(nb);
          nPlayer.push(b);
          nG.push(g + 1);
          nParent.push(s);
          nMoves.push(this.walk(player, stance, reg.parent) + CH[d]);
          heapPush([prio(g + 1, h), h, seq++, t]);
        }
      }
    }
    return { moves: null, player: startPlayer, expansions };
  }
}

/** Shared deterministic expansion budget across a whole level's attempts. */
let budgetLeft = EXPANSIONS_GLOBAL;

/**
 * Solve one room push-optimally. Tries exact A* first (with the exit rule when needed), then
 * weighted A* and greedy best-first as fallbacks. Returns the room's moves, the player's final
 * cell, and the boxes' final positions — or null.
 */
function solveRoom(
  searcher: RoomSearch,
  room: Room,
  needsExit: boolean,
): { moves: string; player: number; boxes: number[] } | null {
  const startBoxes = room.boxes.slice();
  const attempts: { weight: number; greedy: boolean }[] = [
    { weight: 1, greedy: false },
    { weight: 3, greedy: false },
    { weight: 1, greedy: true },
  ];
  for (const { weight, greedy } of attempts) {
    const budget = Math.min(EXPANSIONS_PER_ATTEMPT, budgetLeft);
    if (budget <= 0) break;
    // Phase 1: no exit requirement; accept it when the player can still reach the door.
    const free = searcher.search(startBoxes, room.entry, false, weight, greedy, budget);
    budgetLeft -= free.expansions;
    if (free.moves === null) continue;
    const finalBoxes = finalBoxPositions(searcher, startBoxes, free.moves, room.entry);
    if (finalBoxes === null || !finalBoxes.every((b) => searcher.goalMask[b] === 1)) continue;
    if (!needsExit || canReachDoor(searcher, finalBoxes, free.player, room)) {
      return { moves: free.moves, player: free.player, boxes: finalBoxes };
    }
    // The optimum cannot exit: re-search with the exit required, same weight.
    const budget2 = Math.min(EXPANSIONS_PER_ATTEMPT, budgetLeft);
    if (budget2 <= 0) continue;
    const withExit = searcher.search(startBoxes, room.entry, true, weight, greedy, budget2);
    budgetLeft -= withExit.expansions;
    if (withExit.moves === null) continue;
    const boxes2 = finalBoxPositions(searcher, startBoxes, withExit.moves, room.entry);
    if (boxes2 !== null && boxes2.every((b) => searcher.goalMask[b] === 1)) {
      return { moves: withExit.moves, player: withExit.player, boxes: boxes2 };
    }
  }
  return null;
}

/** Replay a room's move string from its entry cell and return the final box positions. */
function finalBoxPositions(
  searcher: RoomSearch,
  startBoxes: number[],
  moves: string,
  startPlayer: number,
): number[] | null {
  const { w } = searcher;
  const boxAt = new Set(startBoxes);
  let player = startPlayer;
  for (const ch of moves) {
    const d = CH.indexOf(ch);
    if (d < 0) return null;
    const next = player + DY[d] * w + DX[d];
    if (searcher.allowed[next] !== 1) return null;
    if (boxAt.has(next)) {
      const beyond = next + DY[d] * w + DX[d];
      if (searcher.allowed[beyond] !== 1 || boxAt.has(beyond)) return null;
      boxAt.delete(next);
      boxAt.add(beyond);
    }
    player = next;
  }
  return [...boxAt].sort((a, b) => a - b);
}

/** True when the player, standing at `player`, can walk to the room's door. */
function canReachDoor(searcher: RoomSearch, boxes: number[], player: number, room: Room): boolean {
  const { w, h } = searcher.board;
  const boxSet = new Set(boxes);
  const seen = new Uint8Array(w * h);
  const q = [player];
  seen[player] = 1;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    if (c === room.door) return true;
    const x = c % w;
    const y = Math.floor(c / w);
    for (let d = 0; d < 4; d++) {
      const nx = x + DX[d];
      const ny = y + DY[d];
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const n = ny * w + nx;
      if (searcher.allowed[n] === 0 || boxSet.has(n) || seen[n]) continue;
      seen[n] = 1;
      q.push(n);
    }
  }
  return false;
}

/** Walk moves from `from` to `to` over floor cells avoiding `boxes` (whole level). */
function walkWhole(board: Board, boxes: number[], from: number, to: number): string | null {
  const { w, h, wall } = board;
  if (from === to) return '';
  const boxSet = new Set(boxes);
  const parent = new Int32Array(w * h).fill(-2);
  const q = [from];
  parent[from] = -1;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const x = c % w;
    const y = Math.floor(c / w);
    for (let d = 0; d < 4; d++) {
      const nx = x + DX[d];
      const ny = y + DY[d];
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const n = ny * w + nx;
      if (wall[n] === 1 || boxSet.has(n) || parent[n] !== -2) continue;
      parent[n] = c;
      if (n === to) {
        const path: number[] = [];
        for (let t = n; t !== from && t >= 0; t = parent[t]!) path.push(t);
        path.reverse();
        let moves = '';
        let cur = from;
        for (const cell of path) {
          const dx = (cell % w) - (cur % w);
          const dy = Math.floor(cell / w) - Math.floor(cur / w);
          moves += dx === 1 ? 'r' : dx === -1 ? 'l' : dy === 1 ? 'd' : 'u';
          cur = cell;
        }
        return moves;
      }
      q.push(n);
    }
  }
  return null;
}

/** Solve a level by decomposing it into rooms and stitching the rooms together. */
function solveLevel(board: Board): string {
  budgetLeft = EXPANSIONS_GLOBAL;
  const rooms = decompose(board);
  if (rooms !== null) {
    let moves = '';
    let player = board.player;
    let boxes = board.boxes.slice();
    for (let i = 0; i < rooms.length; i++) {
      const room = rooms[i]!;
      const needsExit = i < rooms.length - 1;
      const searcher = new RoomSearch(board, room);
      const solved = solveRoom(searcher, room, needsExit);
      if (solved === null) return '';
      const walk = walkWhole(board, boxes, player, room.entry);
      if (walk === null) return '';
      moves += walk + solved.moves;
      player = solved.player;
      // Replace this room's boxes with their solved positions in the global box list.
      boxes = boxes.filter((b) => room.mask[b] !== 1).concat(solved.boxes);
      boxes.sort((a, b) => a - b);
    }
    return moves;
  }
  // Fallback for levels without the generated shape: one room over the whole board.
  const whole: Room = {
    cells: allFloor(board),
    mask: wholeMask(board),
    door: board.player,
    entry: board.player,
    boxes: board.boxes.slice(),
    goals: allGoals(board),
  };
  const searcher = new RoomSearch(board, whole);
  const solved = solveRoom(searcher, whole, false);
  return solved === null ? '' : solved.moves;
}

function allFloor(board: Board): number[] {
  const out: number[] = [];
  for (let i = 0; i < board.w * board.h; i++) if (board.wall[i] === 0) out.push(i);
  return out;
}

function wholeMask(board: Board): Uint8Array {
  const m = new Uint8Array(board.w * board.h);
  for (let i = 0; i < m.length; i++) if (board.wall[i] === 0) m[i] = 1;
  return m;
}

function allGoals(board: Board): number[] {
  const out: number[] = [];
  for (let i = 0; i < board.w * board.h; i++) if (board.goal[i] === 1) out.push(i);
  return out;
}

// ---------- driver ----------
for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(parseLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8')));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
