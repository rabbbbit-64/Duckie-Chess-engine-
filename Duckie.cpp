// =============================================================================
//  Duckie - a UCI chess engine in C++20
// =============================================================================
//  Layout of this file (top to bottom):
//     1. Basic types, constants, bit helpers, PRNG
//     2. Attack tables      (leaper tables + magic bitboards for sliders)
//     3. Zobrist hashing
//     4. Evaluation tables  (tapered material + piece-square tables)
//     5. Move encoding and move list
//     6. Position           (FEN, make/unmake, attack + legality queries)
//     7. Move generation    (pseudo-legal; Position::isLegal filters)
//     8. Perft + built-in self-test suite
//     9. Transposition table
//    10. Search             (iterative deepening, PVS alpha-beta, quiescence...)
//    11. UCI front-end      (search runs on a worker thread so `stop` works)
//    12. main()
//
//  Build:
//     g++ -std=c++20 -O3 -march=native -DNDEBUG -pthread Duckie.cpp -o Duckie
//
//  Run:
//     ./duckie                 # self-test (quick perft suite), then UCI loop
//     ./duckie --no-selftest    # skip the start-up self-test (e.g. for a GUI)
//     ./duckie --full           # deep perft suite (~1 billion nodes), then exit
// =============================================================================
 
#include <algorithm>
#include <atomic>
#include <bit>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
 
using U64  = std::uint64_t;
using Move = std::uint16_t;
 
// ----------------------------------------------------------------------------
// Thread-safe line output (the search thread and the UCI thread both print).
// ----------------------------------------------------------------------------
static std::mutex g_outMutex;
static void out(const std::string& line) {
    std::lock_guard<std::mutex> lock(g_outMutex);
    std::cout << line << '\n' << std::flush;
}
 
// =============================================================================
// 1. Basic types, constants and bit helpers
// =============================================================================
constexpr int WHITE = 0, BLACK = 1;
constexpr int PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5;
constexpr int NO_PIECE = 15;   // a piece is (colour << 3) | type  -> 0..5 and 8..13
constexpr int NO_SQ    = 64;
 
constexpr int MAX_PLY     = 128;
constexpr int MATE        = 32000;
constexpr int INF         = 32001;
constexpr int MATE_IN_MAX = MATE - MAX_PLY;
 
constexpr const char* START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
 
// Little-endian rank-file mapping: a1 = 0, b1 = 1, ... h8 = 63.
enum Square : int {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8
};
 
enum CastlingRight : int { WK = 1, WQ = 2, BK = 4, BQ = 8 };
 
constexpr U64 FILE_A = 0x0101010101010101ULL;
constexpr U64 FILE_H = FILE_A << 7;
constexpr U64 RANK_1 = 0xFFULL;
constexpr U64 RANK_3 = RANK_1 << 16;
constexpr U64 RANK_6 = RANK_1 << 40;
constexpr U64 RANK_8 = RANK_1 << 56;
 
constexpr int  makePiece(int c, int pt) { return (c << 3) | pt; }
constexpr int  typeOf(int pc)           { return pc & 7; }
constexpr int  colorOf(int pc)          { return pc >> 3; }
constexpr U64  bit(int sq)              { return 1ULL << sq; }
inline int     popcount(U64 b)          { return std::popcount(b); }
inline int     lsb(U64 b)               { return std::countr_zero(b); }
inline int     popLsb(U64& b)           { const int s = lsb(b); b &= b - 1; return s; }
 
constexpr std::string_view PIECE_CHARS = "PNBRQK  pnbrqk";
 
// xorshift64* -- small, fast, deterministic PRNG (magics + Zobrist keys).
struct Rng {
    U64 s;
    explicit Rng(U64 seed) : s(seed) {}
    U64 next() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 0x2545F4914F6CDD1DULL;
    }
    U64 sparse() { return next() & next() & next(); }   // few set bits: good magic candidates
};
 
// =============================================================================
// 2. Attack tables
// =============================================================================
namespace Attacks {
 
U64 pawn[2][64];     // pawn[c][sq]  = squares attacked by a pawn of colour c standing on sq
U64 knight[64];
U64 king[64];
 
struct Magic {
    U64  mask  = 0;
    U64  magic = 0;
    U64* table = nullptr;
    int  shift = 0;
};
constexpr std::size_t ROOK_TABLE_SIZE = 102400, BISHOP_TABLE_SIZE = 5248;
Magic rookMagic[64], bishopMagic[64];
U64   rookTable[ROOK_TABLE_SIZE], bishopTable[BISHOP_TABLE_SIZE];
 
inline U64 rook(int sq, U64 occ) {
    const Magic& m = rookMagic[sq];
    return m.table[((occ & m.mask) * m.magic) >> m.shift];
}
inline U64 bishop(int sq, U64 occ) {
    const Magic& m = bishopMagic[sq];
    return m.table[((occ & m.mask) * m.magic) >> m.shift];
}
inline U64 queen(int sq, U64 occ) { return rook(sq, occ) | bishop(sq, occ); }
 
// Slow reference ray-walker, used only while building the magic tables.
static U64 slide(int sq, U64 occ, bool diagonal) {
    static constexpr int DR[2][4] = {{1, -1, 0, 0}, {1, 1, -1, -1}};
    static constexpr int DF[2][4] = {{0, 0, 1, -1}, {1, -1, 1, -1}};
    U64 att = 0;
    for (int d = 0; d < 4; ++d) {
        const int dr = DR[diagonal][d], df = DF[diagonal][d];
        for (int r = (sq >> 3) + dr, f = (sq & 7) + df; r >= 0 && r < 8 && f >= 0 && f < 8; r += dr, f += df) {
            att |= bit(r * 8 + f);
            if (occ & bit(r * 8 + f)) break;
        }
    }
    return att;
}
 
// Finds a magic multiplier for every square by random search ("fancy" magics:
// every square gets a table exactly as large as its number of relevant subsets).
static void initMagics(Magic* magics, U64* table, std::size_t tableSize, bool diagonal) {
    static const U64 seeds[8] = {8977, 44560, 54343, 38998, 5731, 95205, 104912, 17020};
    static U64 occupancy[4096], reference[4096];
    static int epoch[4096];
    std::size_t offset = 0;
 
    for (int sq = 0; sq < 64; ++sq) {
        const U64 edges = ((RANK_1 | RANK_8) & ~(RANK_1 << (sq & ~7))) |
                          ((FILE_A | FILE_H) & ~(FILE_A << (sq & 7)));
        Magic& m = magics[sq];
        m.mask  = slide(sq, 0, diagonal) & ~edges;
        m.shift = 64 - popcount(m.mask);
        m.table = table + offset;
 
        // Enumerate every subset of the mask (Carry-Rippler) with its true attack set.
        int n = 0;
        U64 b = 0;
        do {
            occupancy[n] = b;
            reference[n] = slide(sq, b, diagonal);
            ++n;
            b = (b - m.mask) & m.mask;
        } while (b);
        offset += static_cast<std::size_t>(n);
 
        Rng rng(seeds[sq >> 3]);
        std::fill(epoch, epoch + n, 0);
        for (int attempt = 1;; ++attempt) {
            const U64 cand = rng.sparse();
            if (popcount((m.mask * cand) >> 56) < 6) { --attempt; continue; }
            bool ok = true;
            for (int i = 0; i < n && ok; ++i) {
                const int idx = static_cast<int>((occupancy[i] * cand) >> m.shift);
                if (epoch[idx] != attempt) { epoch[idx] = attempt; m.table[idx] = reference[i]; }
                else if (m.table[idx] != reference[i]) ok = false;   // destructive collision
            }
            if (ok) { m.magic = cand; break; }
        }
    }
    if (offset > tableSize) { std::cerr << "magic table overflow\n"; std::abort(); }
}
 
static void initLeapers() {
    static const int KN[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    static const int KG[8][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
    for (int sq = 0; sq < 64; ++sq) {
        const U64 b = bit(sq);
        pawn[WHITE][sq] = ((b & ~FILE_A) << 7) | ((b & ~FILE_H) << 9);
        pawn[BLACK][sq] = ((b & ~FILE_A) >> 9) | ((b & ~FILE_H) >> 7);
        knight[sq] = king[sq] = 0;
        for (const auto& d : KN) {
            const int r = (sq >> 3) + d[0], f = (sq & 7) + d[1];
            if (r >= 0 && r < 8 && f >= 0 && f < 8) knight[sq] |= bit(r * 8 + f);
        }
        for (const auto& d : KG) {
            const int r = (sq >> 3) + d[0], f = (sq & 7) + d[1];
            if (r >= 0 && r < 8 && f >= 0 && f < 8) king[sq] |= bit(r * 8 + f);
        }
    }
}
 
void init() {
    initLeapers();
    initMagics(rookMagic, rookTable, ROOK_TABLE_SIZE, false);
    initMagics(bishopMagic, bishopTable, BISHOP_TABLE_SIZE, true);
}
 
}  // namespace Attacks
 
// =============================================================================
// 3. Zobrist hashing
// =============================================================================
namespace Zobrist {
U64 psq[16][64];   // [piece][square]
U64 side;          // XORed in when Black is to move
U64 castle[16];    // indexed by the 4-bit castling-rights mask
U64 ep[8];         // indexed by en-passant file
 
void init() {
    Rng rng(0x9E3779B97F4A7C15ULL);
    for (auto& row : psq) for (auto& k : row) k = rng.next();
    side = rng.next();
    for (auto& k : castle) k = rng.next();
    for (auto& k : ep) k = rng.next();
}
}  // namespace Zobrist
 
// Rights that survive a move touching the given square (applied for both from and to).
static int g_castleMask[64];
static void initCastleMask() {
    for (int& m : g_castleMask) m = 0;
    g_castleMask[E1] = WK | WQ; g_castleMask[A1] = WQ; g_castleMask[H1] = WK;
    g_castleMask[E8] = BK | BQ; g_castleMask[A8] = BQ; g_castleMask[H8] = BK;
}
 
// =============================================================================
// 4. Evaluation tables: tapered material + piece-square tables
// =============================================================================
// Tables are in the classic "Simplified Evaluation Function" orientation:
// written from White's point of view with rank 8 on the first row.
namespace Eval {
 
constexpr int MAT_MG[6] = {100, 320, 330, 500, 900, 0};
constexpr int MAT_EG[6] = {120, 300, 320, 520, 920, 0};
constexpr int PHASE[8]  = {0, 1, 1, 2, 4, 0, 0, 0};   // total = 24 at the start
constexpr int PHASE_TOTAL = 24;
 
constexpr int PST_PAWN_MG[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0};
constexpr int PST_PAWN_EG[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
     80,  80,  80,  80,  80,  80,  80,  80,
     50,  50,  50,  50,  50,  50,  50,  50,
     30,  30,  30,  30,  30,  30,  30,  30,
     20,  20,  20,  20,  20,  20,  20,  20,
     10,  10,  10,  10,  10,  10,  10,  10,
     10,  10,  10,  10,  10,  10,  10,  10,
      0,   0,   0,   0,   0,   0,   0,   0};
constexpr int PST_KNIGHT[64] = {
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50};
constexpr int PST_BISHOP[64] = {
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20};
constexpr int PST_ROOK[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0};
constexpr int PST_QUEEN[64] = {
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20};
constexpr int PST_KING_MG[64] = {
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20};
constexpr int PST_KING_EG[64] = {
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50};
 
// Combined (material + PST) scores from White's point of view, indexed [piece][square].
// Black pieces carry a negative sign, so a position's score is a plain sum.
int mg[16][64], eg[16][64];
 
void init() {
    const int* const tmg[6] = {PST_PAWN_MG, PST_KNIGHT, PST_BISHOP, PST_ROOK, PST_QUEEN, PST_KING_MG};
    const int* const teg[6] = {PST_PAWN_EG, PST_KNIGHT, PST_BISHOP, PST_ROOK, PST_QUEEN, PST_KING_EG};
    for (int pt = 0; pt < 6; ++pt)
        for (int sq = 0; sq < 64; ++sq) {
            mg[makePiece(WHITE, pt)][sq] =  MAT_MG[pt] + tmg[pt][sq ^ 56];
            eg[makePiece(WHITE, pt)][sq] =  MAT_EG[pt] + teg[pt][sq ^ 56];
            mg[makePiece(BLACK, pt)][sq] = -(MAT_MG[pt] + tmg[pt][sq]);
            eg[makePiece(BLACK, pt)][sq] = -(MAT_EG[pt] + teg[pt][sq]);
        }
}
}  // namespace Eval
 
// =============================================================================
// 5. Move encoding and move list
// =============================================================================
// 16-bit move: bits 0-5 from, 6-11 to, 12-15 flags.
//   flag bit 2 (4) = capture, flag bit 3 (8) = promotion, low 2 bits of a
//   promotion flag select the piece (0 = N, 1 = B, 2 = R, 3 = Q).
enum MoveFlag : int {
    QUIET = 0, DOUBLE_PUSH = 1, KING_CASTLE = 2, QUEEN_CASTLE = 3,
    CAPTURE = 4, EP_CAPTURE = 5,
    PROMO_N = 8, PROMO_B = 9, PROMO_R = 10, PROMO_Q = 11,
    PROMO_CAP_N = 12, PROMO_CAP_B = 13, PROMO_CAP_R = 14, PROMO_CAP_Q = 15
};
constexpr Move NO_MOVE = 0;
 
constexpr Move encodeMove(int from, int to, int flags) { return static_cast<Move>(from | (to << 6) | (flags << 12)); }
constexpr int  moveFrom(Move m)  { return m & 63; }
constexpr int  moveTo(Move m)    { return (m >> 6) & 63; }
constexpr int  moveFlags(Move m) { return m >> 12; }
constexpr bool isCapture(Move m) { return moveFlags(m) & 4; }
constexpr bool isPromo(Move m)   { return moveFlags(m) & 8; }
constexpr int  promoType(Move m) { return (moveFlags(m) & 3) + KNIGHT; }
 
static std::string moveToString(Move m) {
    if (m == NO_MOVE) return "0000";
    std::string s;
    s += static_cast<char>('a' + (moveFrom(m) & 7));
    s += static_cast<char>('1' + (moveFrom(m) >> 3));
    s += static_cast<char>('a' + (moveTo(m) & 7));
    s += static_cast<char>('1' + (moveTo(m) >> 3));
    if (isPromo(m)) s += "nbrq"[moveFlags(m) & 3];
    return s;
}
 
struct MoveList {
    Move moves[256];
    int  count = 0;
    void add(int from, int to, int flags) { moves[count++] = encodeMove(from, to, flags); }
};
 
// =============================================================================
// 6. Position
// =============================================================================
struct StateInfo {          // everything make/unmake cannot recompute cheaply
    U64 key;
    int castling;
    int ep;
    int halfmove;
    int captured;
};
 
class Position {
public:
    U64 byType[8]  = {};        // bitboard per piece type (both colours)
    U64 byColor[2] = {};
    U64 occ        = 0;
    std::uint8_t board[64];     // mailbox: piece on each square or NO_PIECE
    int stm = WHITE, castling = 0, ep = NO_SQ, halfmove = 0, fullmove = 1;
    U64 key = 0;                // Zobrist hash, updated incrementally
    int mg = 0, eg = 0, phase = 0;   // incremental evaluation terms (White POV)
    std::vector<StateInfo> history;  // one entry per move made (also holds game history)
 
    Position() { std::memset(board, NO_PIECE, sizeof board); history.reserve(512); }
 
    U64 pieces(int c, int pt) const { return byType[pt] & byColor[c]; }
    int kingSq(int c) const { return lsb(pieces(c, KING)); }
 
    bool setFen(const std::string& fen);
    std::string fen() const;
    std::string toString() const;
 
    void makeMove(Move m);
    void unmakeMove(Move m);
    void makeNull();
    void unmakeNull();
 
    U64  attackersTo(int sq, U64 o) const;
    bool attacked(int sq, int by) const { return attackersTo(sq, occ) & byColor[by]; }
    bool inCheck() const { return attacked(kingSq(stm), stm ^ 1); }
    bool isLegal(Move m) const;                 // m must be pseudo-legal in this position
    bool isRepetition() const;
    bool insufficientMaterial() const;
    bool hasNonPawnMaterial(int c) const {
        return (pieces(c, KNIGHT) | pieces(c, BISHOP) | pieces(c, ROOK) | pieces(c, QUEEN)) != 0;
    }
    bool verify() const;                        // recompute everything from scratch and compare
 
private:
    void put(int pc, int sq);
    void remove(int pc, int sq);
};
 
// Add / remove a piece, keeping bitboards, mailbox, hash and eval in sync.
inline void Position::put(int pc, int sq) {
    const U64 b = bit(sq);
    byType[typeOf(pc)] |= b; byColor[colorOf(pc)] |= b; occ |= b;
    board[sq] = static_cast<std::uint8_t>(pc);
    key ^= Zobrist::psq[pc][sq];
    mg += Eval::mg[pc][sq]; eg += Eval::eg[pc][sq]; phase += Eval::PHASE[typeOf(pc)];
}
inline void Position::remove(int pc, int sq) {
    const U64 b = bit(sq);
    byType[typeOf(pc)] ^= b; byColor[colorOf(pc)] ^= b; occ ^= b;
    board[sq] = NO_PIECE;
    key ^= Zobrist::psq[pc][sq];
    mg -= Eval::mg[pc][sq]; eg -= Eval::eg[pc][sq]; phase -= Eval::PHASE[typeOf(pc)];
}
 
inline U64 Position::attackersTo(int sq, U64 o) const {
    return (Attacks::pawn[BLACK][sq] & pieces(WHITE, PAWN)) |
           (Attacks::pawn[WHITE][sq] & pieces(BLACK, PAWN)) |
           (Attacks::knight[sq] & byType[KNIGHT]) |
           (Attacks::king[sq] & byType[KING]) |
           (Attacks::bishop(sq, o) & (byType[BISHOP] | byType[QUEEN])) |
           (Attacks::rook(sq, o) & (byType[ROOK] | byType[QUEEN]));
}
 
bool Position::setFen(const std::string& fenStr) {
    std::istringstream ss(fenStr);
    std::string boardStr, side, castleStr = "-", epStr = "-";
    int hm = 0, fm = 1;
    if (!(ss >> boardStr >> side)) return false;
    ss >> castleStr >> epStr >> hm >> fm;
 
    byType[0] = byType[1] = byType[2] = byType[3] = byType[4] = byType[5] = byType[6] = byType[7] = 0;
    byColor[0] = byColor[1] = occ = 0;
    std::memset(board, NO_PIECE, sizeof board);
    key = 0; mg = eg = phase = 0;
    castling = 0; ep = NO_SQ;
    history.clear();
 
    int sq = A8;
    for (const char ch : boardStr) {
        if (ch == '/') { sq -= 16; continue; }
        if (std::isdigit(static_cast<unsigned char>(ch))) { sq += ch - '0'; continue; }
        const std::size_t pt = std::string_view("PNBRQK").find(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
        if (pt == std::string_view::npos || sq < 0 || sq > 63) return false;
        put(makePiece(std::islower(static_cast<unsigned char>(ch)) ? BLACK : WHITE, static_cast<int>(pt)), sq++);
    }
    if (popcount(pieces(WHITE, KING)) != 1 || popcount(pieces(BLACK, KING)) != 1) return false;
    if (byType[PAWN] & (RANK_1 | RANK_8)) return false;
 
    stm = (side == "b") ? BLACK : WHITE;
    if (stm == BLACK) key ^= Zobrist::side;
 
    for (const char ch : castleStr) {
        if (ch == 'K') castling |= WK;
        else if (ch == 'Q') castling |= WQ;
        else if (ch == 'k') castling |= BK;
        else if (ch == 'q') castling |= BQ;
    }
    // Drop rights that do not match the placement of king and rooks.
    if (board[E1] != makePiece(WHITE, KING)) castling &= ~(WK | WQ);
    if (board[H1] != makePiece(WHITE, ROOK)) castling &= ~WK;
    if (board[A1] != makePiece(WHITE, ROOK)) castling &= ~WQ;
    if (board[E8] != makePiece(BLACK, KING)) castling &= ~(BK | BQ);
    if (board[H8] != makePiece(BLACK, ROOK)) castling &= ~BK;
    if (board[A8] != makePiece(BLACK, ROOK)) castling &= ~BQ;
    key ^= Zobrist::castle[castling];
 
    // Keep the e.p. square only if a pawn could really capture on it
    // (this keeps hash keys of equal positions identical).
    if (epStr.size() == 2 && epStr[0] >= 'a' && epStr[0] <= 'h' && (epStr[1] == '3' || epStr[1] == '6')) {
        const int eps = (epStr[1] - '1') * 8 + (epStr[0] - 'a');
        const int them = stm ^ 1;
        if ((eps >> 3) == (stm == WHITE ? 5 : 2) && board[eps] == NO_PIECE &&
            board[eps ^ 8] == makePiece(them, PAWN) && (Attacks::pawn[them][eps] & pieces(stm, PAWN))) {
            ep = eps;
            key ^= Zobrist::ep[eps & 7];
        }
    }
    halfmove = std::max(hm, 0);
    fullmove = std::max(fm, 1);
 
    // The side that just moved must not leave its king capturable.
    if (attacked(kingSq(stm ^ 1), stm)) return false;
    return true;
}
 
std::string Position::fen() const {
    std::ostringstream ss;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            const int pc = board[r * 8 + f];
            if (pc == NO_PIECE) { ++empty; continue; }
            if (empty) { ss << empty; empty = 0; }
            ss << PIECE_CHARS[pc];
        }
        if (empty) ss << empty;
        if (r) ss << '/';
    }
    ss << (stm == WHITE ? " w " : " b ");
    if (!castling) ss << '-';
    if (castling & WK) ss << 'K';
    if (castling & WQ) ss << 'Q';
    if (castling & BK) ss << 'k';
    if (castling & BQ) ss << 'q';
    ss << ' ';
    if (ep == NO_SQ) ss << '-';
    else ss << static_cast<char>('a' + (ep & 7)) << static_cast<char>('1' + (ep >> 3));
    ss << ' ' << halfmove << ' ' << fullmove;
    return ss.str();
}
 
std::string Position::toString() const {
    std::ostringstream ss;
    ss << "\n +---+---+---+---+---+---+---+---+\n";
    for (int r = 7; r >= 0; --r) {
        for (int f = 0; f < 8; ++f) {
            const int pc = board[r * 8 + f];
            ss << " | " << (pc == NO_PIECE ? ' ' : PIECE_CHARS[pc]);
        }
        ss << " | " << (r + 1) << "\n +---+---+---+---+---+---+---+---+\n";
    }
    ss << "   a   b   c   d   e   f   g   h\n\nFEN: " << fen() << "\nKey: " << std::hex << key << std::dec;
    return ss.str();
}
 
void Position::makeMove(Move m) {
    const int from = moveFrom(m), to = moveTo(m), fl = moveFlags(m);
    const int us = stm, them = us ^ 1;
    const int pc = board[from];
 
    history.push_back(StateInfo{key, castling, ep, halfmove, NO_PIECE});
    StateInfo& st = history.back();
 
    if (ep != NO_SQ) key ^= Zobrist::ep[ep & 7];
    key ^= Zobrist::castle[castling];
    ep = NO_SQ;
    ++halfmove;
 
    if (fl == EP_CAPTURE) {                     // victim sits behind the target square
        remove(makePiece(them, PAWN), to ^ 8);
        halfmove = 0;
    } else if (fl & 4) {                        // ordinary capture or promotion-capture
        st.captured = board[to];
        remove(st.captured, to);
        halfmove = 0;
    }
 
    remove(pc, from);
    put((fl & 8) ? makePiece(us, promoType(m)) : pc, to);
 
    if (fl == KING_CASTLE) {
        const int r = makePiece(us, ROOK);
        remove(r, to + 1); put(r, to - 1);
    } else if (fl == QUEEN_CASTLE) {
        const int r = makePiece(us, ROOK);
        remove(r, to - 2); put(r, to + 1);
    } else if (typeOf(pc) == PAWN) {
        halfmove = 0;
        if (fl == DOUBLE_PUSH) {
            const int eps = (from + to) >> 1;
            if (Attacks::pawn[us][eps] & pieces(them, PAWN)) { ep = eps; key ^= Zobrist::ep[eps & 7]; }
        }
    }
 
    castling &= ~(g_castleMask[from] | g_castleMask[to]);
    key ^= Zobrist::castle[castling];
    key ^= Zobrist::side;
    stm = them;
    if (us == BLACK) ++fullmove;
}
 
void Position::unmakeMove(Move m) {
    const int from = moveFrom(m), to = moveTo(m), fl = moveFlags(m);
    stm ^= 1;
    const int us = stm, them = us ^ 1;
    const StateInfo& st = history.back();
 
    if (fl & 8) { remove(board[to], to); put(makePiece(us, PAWN), from); }
    else        { const int pc = board[to]; remove(pc, to); put(pc, from); }
 
    if (fl == KING_CASTLE) {
        const int r = makePiece(us, ROOK);
        remove(r, to - 1); put(r, to + 1);
    } else if (fl == QUEEN_CASTLE) {
        const int r = makePiece(us, ROOK);
        remove(r, to + 1); put(r, to - 2);
    }
 
    if (fl == EP_CAPTURE) put(makePiece(them, PAWN), to ^ 8);
    else if (fl & 4)      put(st.captured, to);
 
    castling = st.castling; ep = st.ep; halfmove = st.halfmove; key = st.key;
    if (us == BLACK) --fullmove;
    history.pop_back();
}
 
// A "null move" just passes the turn (used by null-move pruning).
// halfmove is zeroed so repetition scans never look back across it.
void Position::makeNull() {
    history.push_back(StateInfo{key, castling, ep, halfmove, NO_PIECE});
    if (ep != NO_SQ) { key ^= Zobrist::ep[ep & 7]; ep = NO_SQ; }
    key ^= Zobrist::side;
    stm ^= 1;
    halfmove = 0;
}
void Position::unmakeNull() {
    const StateInfo& st = history.back();
    stm ^= 1; ep = st.ep; halfmove = st.halfmove; key = st.key;
    history.pop_back();
}
 
// Legality test for a pseudo-legal move without playing it: build the occupancy
// after the move and ask whether the (new) king square is attacked by anything
// that is not captured by the move. Handles pins, discovered checks, e.p. and
// king moves in one uniform way.
bool Position::isLegal(Move m) const {
    const int from = moveFrom(m), to = moveTo(m), fl = moveFlags(m);
    const int us = stm, them = us ^ 1;
    U64 occ2 = (occ ^ bit(from)) | bit(to);
    U64 removed = 0;
    if (fl == EP_CAPTURE) { removed = bit(to ^ 8); occ2 ^= removed; }
    else if (fl & 4)      removed = bit(to);
    const int ksq = (typeOf(board[from]) == KING) ? to : lsb(pieces(us, KING));
    return !(attackersTo(ksq, occ2) & byColor[them] & ~removed);
}
 
// True if the current position already occurred since the last irreversible move.
bool Position::isRepetition() const {
    const int n = static_cast<int>(history.size());
    const int limit = std::min(halfmove, n);
    for (int i = 2; i <= limit; i += 2)
        if (history[n - i].key == key) return true;
    return false;
}
 
bool Position::insufficientMaterial() const {
    if (byType[PAWN] | byType[ROOK] | byType[QUEEN]) return false;
    return popcount(byType[KNIGHT] | byType[BISHOP]) <= 1;   // K v K, K+minor v K
}
 
// Debug aid: rebuild hash / eval / bitboards from the mailbox and compare.
bool Position::verify() const {
    U64 k = 0, bb[2] = {0, 0}, bt[8] = {};
    int m = 0, e = 0, ph = 0;
    for (int sq = 0; sq < 64; ++sq) {
        const int pc = board[sq];
        if (pc == NO_PIECE) continue;
        bb[colorOf(pc)] |= bit(sq); bt[typeOf(pc)] |= bit(sq);
        k ^= Zobrist::psq[pc][sq];
        m += Eval::mg[pc][sq]; e += Eval::eg[pc][sq]; ph += Eval::PHASE[typeOf(pc)];
    }
    if (stm == BLACK) k ^= Zobrist::side;
    k ^= Zobrist::castle[castling];
    if (ep != NO_SQ) k ^= Zobrist::ep[ep & 7];
    for (int i = 0; i < 8; ++i) if (bt[i] != byType[i]) return false;
    return bb[0] == byColor[0] && bb[1] == byColor[1] && occ == (bb[0] | bb[1]) &&
           k == key && m == mg && e == eg && ph == phase;
}
 
// =============================================================================
// 7. Move generation (pseudo-legal)
// =============================================================================
template <int PT>
inline U64 pieceAttacks(int sq, U64 occ) {
    if constexpr (PT == KNIGHT)      return Attacks::knight[sq];
    else if constexpr (PT == BISHOP) return Attacks::bishop(sq, occ);
    else if constexpr (PT == ROOK)   return Attacks::rook(sq, occ);
    else if constexpr (PT == QUEEN)  return Attacks::queen(sq, occ);
    else                             return Attacks::king[sq];
}
 
template <int PT, bool CapsOnly>
static void genPieceMoves(const Position& pos, MoveList& ml, U64 targets, U64 theirs) {
    for (U64 b = pos.pieces(pos.stm, PT); b;) {
        const int from = popLsb(b);
        const U64 att = pieceAttacks<PT>(from, pos.occ) & targets;
        for (U64 c = att & theirs; c;) ml.add(from, popLsb(c), CAPTURE);
        if constexpr (!CapsOnly)
            for (U64 q = att & ~theirs; q;) ml.add(from, popLsb(q), QUIET);
    }
}
 
// CapsOnly = true  -> captures and queen promotions only (quiescence search)
// CapsOnly = false -> every pseudo-legal move (the king may still be left in check)
template <bool CapsOnly>
void generateMoves(const Position& pos, MoveList& ml) {
    const int us = pos.stm, them = us ^ 1;
    const U64 ours = pos.byColor[us], theirs = pos.byColor[them], occ = pos.occ;
    const U64 pawns = pos.pieces(us, PAWN);
    const U64 promoRank = (us == WHITE) ? RANK_8 : RANK_1 << 0 * 0 | (RANK_1);   // placeholder, fixed below
    (void)promoRank;
    const U64 lastRank = (us == WHITE) ? RANK_8 : RANK_1;
 
    auto addPawn = [&](int from, int to, bool cap) {
        if (bit(to) & lastRank) {
            const int base = cap ? PROMO_CAP_N : PROMO_N;
            ml.add(from, to, base + 3);                         // queen
            if constexpr (!CapsOnly) {
                ml.add(from, to, base + 2);                     // rook
                ml.add(from, to, base + 1);                     // bishop
                ml.add(from, to, base);                         // knight
            }
        } else {
            ml.add(from, to, cap ? CAPTURE : QUIET);
        }
    };
 
    // ---- pawn pushes -------------------------------------------------------
    const int push = (us == WHITE) ? 8 : -8;
    U64 one = ((us == WHITE) ? pawns << 8 : pawns >> 8) & ~occ;
    const U64 two = ((us == WHITE) ? ((one & RANK_3) << 8) : ((one & RANK_6) >> 8)) & ~occ;
    if constexpr (CapsOnly) one &= lastRank;
    for (U64 b = one; b;) { const int to = popLsb(b); addPawn(to - push, to, false); }
    if constexpr (!CapsOnly)
        for (U64 b = two; b;) { const int to = popLsb(b); ml.add(to - 2 * push, to, DOUBLE_PUSH); }
 
    // ---- pawn captures + en passant ---------------------------------------
    const U64 capL = ((us == WHITE) ? (pawns & ~FILE_A) << 7 : (pawns & ~FILE_A) >> 9) & theirs;
    const U64 capR = ((us == WHITE) ? (pawns & ~FILE_H) << 9 : (pawns & ~FILE_H) >> 7) & theirs;
    const int dL = (us == WHITE) ? 7 : -9, dR = (us == WHITE) ? 9 : -7;
    for (U64 b = capL; b;) { const int to = popLsb(b); addPawn(to - dL, to, true); }
    for (U64 b = capR; b;) { const int to = popLsb(b); addPawn(to - dR, to, true); }
    if (pos.ep != NO_SQ)
        for (U64 b = Attacks::pawn[them][pos.ep] & pawns; b;) ml.add(popLsb(b), pos.ep, EP_CAPTURE);
 
    // ---- pieces ------------------------------------------------------------
    const U64 targets = CapsOnly ? theirs : ~ours;
    genPieceMoves<KNIGHT, CapsOnly>(pos, ml, targets, theirs);
    genPieceMoves<BISHOP, CapsOnly>(pos, ml, targets, theirs);
    genPieceMoves<ROOK,   CapsOnly>(pos, ml, targets, theirs);
    genPieceMoves<QUEEN,  CapsOnly>(pos, ml, targets, theirs);
    genPieceMoves<KING,   CapsOnly>(pos, ml, targets, theirs);
 
    // ---- castling (squares the king crosses must not be attacked) ----------
    if constexpr (!CapsOnly) {
        if (us == WHITE && (pos.castling & (WK | WQ))) {
            if ((pos.castling & WK) && !(occ & (bit(F1) | bit(G1))) &&
                !pos.attacked(E1, them) && !pos.attacked(F1, them) && !pos.attacked(G1, them))
                ml.add(E1, G1, KING_CASTLE);
            if ((pos.castling & WQ) && !(occ & (bit(B1) | bit(C1) | bit(D1))) &&
                !pos.attacked(E1, them) && !pos.attacked(D1, them) && !pos.attacked(C1, them))
                ml.add(E1, C1, QUEEN_CASTLE);
        } else if (us == BLACK && (pos.castling & (BK | BQ))) {
            if ((pos.castling & BK) && !(occ & (bit(F8) | bit(G8))) &&
                !pos.attacked(E8, them) && !pos.attacked(F8, them) && !pos.attacked(G8, them))
                ml.add(E8, G8, KING_CASTLE);
            if ((pos.castling & BQ) && !(occ & (bit(B8) | bit(C8) | bit(D8))) &&
                !pos.attacked(E8, them) && !pos.attacked(D8, them) && !pos.attacked(C8, them))
                ml.add(E8, C8, QUEEN_CASTLE);
        }
    }
}
 
static Move parseMove(const Position& pos, const std::string& s) {
    MoveList ml;
    generateMoves<false>(pos, ml);
    for (int i = 0; i < ml.count; ++i)
        if (pos.isLegal(ml.moves[i]) && moveToString(ml.moves[i]) == s) return ml.moves[i];
    return NO_MOVE;
}
 
// =============================================================================
// 8. Perft and the built-in self-test suite
// =============================================================================
// Counts leaf nodes of the legal-move tree. At depth 1 the legal moves are just
// counted ("bulk counting"), which saves a make/unmake per leaf.
static U64 perft(Position& pos, int depth) {
    if (depth == 0) return 1;
    MoveList ml;
    generateMoves<false>(pos, ml);
    U64 nodes = 0;
    if (depth == 1) {
        for (int i = 0; i < ml.count; ++i) nodes += pos.isLegal(ml.moves[i]);
        return nodes;
    }
    for (int i = 0; i < ml.count; ++i) {
        const Move m = ml.moves[i];
        if (!pos.isLegal(m)) continue;
        pos.makeMove(m);
        nodes += perft(pos, depth - 1);
        pos.unmakeMove(m);
    }
    return nodes;
}
 
// Walks the tree and checks after every move/unmake that hash, eval terms and
// bitboards are consistent with a from-scratch recomputation.
static bool perftVerify(Position& pos, int depth, U64& visited) {
    if (!pos.verify()) return false;
    ++visited;
    if (depth == 0) return true;
    MoveList ml;
    generateMoves<false>(pos, ml);
    for (int i = 0; i < ml.count; ++i) {
        const Move m = ml.moves[i];
        if (!pos.isLegal(m)) continue;
        const U64 before = pos.key;
        pos.makeMove(m);
        const bool ok = perftVerify(pos, depth - 1, visited);
        pos.unmakeMove(m);
        if (!ok || pos.key != before || !pos.verify()) return false;
    }
    return true;
}
 
struct PerftCase {
    const char* name;
    const char* fen;
    int firstDepth;                 // nodes[0] belongs to this depth
    std::vector<U64> nodes;         // expected node counts for consecutive depths
    int quickDepth;                 // deepest depth in the start-up self-test (0 = full run only)
};
 
// Reference values from the Chess Programming Wiki perft results and the
// well-known "perftsuite" special-case positions.
static const std::vector<PerftCase>& perftSuite() {
    static const std::vector<PerftCase> suite = {
        {"startpos", START_FEN, 1, {20, 400, 8902, 197281, 4865609, 119060324}, 5},
        {"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1,
            {48, 2039, 97862, 4085603, 193690690}, 4},
        {"position 3 (endgame)", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 1,
            {14, 191, 2812, 43238, 674624, 11030083}, 5},
        {"position 4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 1,
            {6, 264, 9467, 422333, 15833292}, 4},
        {"position 4 (mirrored)", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", 1,
            {6, 264, 9467, 422333, 15833292}, 4},
        {"position 5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 1,
            {44, 1486, 62379, 2103487, 89941194}, 4},
        {"position 6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 1,
            {46, 2079, 89890, 3894594, 164075551}, 4},
        // ---- special cases (full run only) ----
        {"illegal e.p. (rank pin)",    "3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1", 6, {1134888}, 0},
        {"illegal e.p. (discovered)",  "8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1", 6, {1015133}, 0},
        {"e.p. captures the checker",  "8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1", 6, {1440467}, 0},
        {"short castle gives check",   "5k2/8/8/8/8/8/8/4K2R w K - 0 1", 6, {661072}, 0},
        {"long castle gives check",    "3k4/8/8/8/8/8/8/R3K3 w Q - 0 1", 6, {803711}, 0},
        {"castling rights / attacks",  "r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1", 4, {1274206}, 0},
        {"castling prevented",         "r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1", 4, {1720476}, 0},
        {"promote out of check",       "2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1", 6, {3821001}, 0},
        {"discovered check",           "8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1", 5, {1004658}, 0},
        {"promote to give check",      "4k3/1P6/8/8/8/8/K7/8 w - - 0 1", 6, {217342}, 0},
        {"under-promote to check",     "8/P1k5/K7/8/8/8/8/8 w - - 0 1", 6, {92683}, 0},
        {"self stalemate",             "K1k5/8/P7/8/8/8/8/8 w - - 0 1", 6, {2217}, 0},
        {"stalemate and checkmate",    "8/k1P5/8/1K6/8/8/8/8 w - - 0 1", 7, {567584}, 0},
        {"double check",               "8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1", 4, {23527}, 0},
    };
    return suite;
}
 
// Runs the perft suite, printing node counts, timing and speed. Returns true
// only if every count matches the published reference value.
static bool runSelfTest(bool full) {
    using Clock = std::chrono::steady_clock;
    out(std::string("info string ===== perft self-test (") + (full ? "full" : "quick") + ") =====");
    bool ok = true;
    U64 totalNodes = 0;
    double totalSec = 0.0;
    Position pos;
 
    for (const PerftCase& c : perftSuite()) {
        if (!full && c.quickDepth == 0) continue;
        if (!pos.setFen(c.fen)) { out(std::string("info string invalid FEN in suite: ") + c.name); return false; }
        const int last = c.firstDepth + static_cast<int>(c.nodes.size()) - 1;
        const int maxDepth = full ? last : std::min(last, c.quickDepth);
        for (int d = c.firstDepth; d <= maxDepth; ++d) {
            const auto t0 = Clock::now();
            const U64 n = perft(pos, d);
            const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
            const U64 expected = c.nodes[static_cast<std::size_t>(d - c.firstDepth)];
            totalNodes += n;
            totalSec += sec;
            std::ostringstream ss;
            ss << "info string perft " << std::left << std::setw(26) << c.name << std::right
               << " depth " << d << std::setw(13) << n << " nodes  " << (n == expected ? "OK  " : "FAIL")
               << std::fixed << std::setprecision(1) << std::setw(9) << sec * 1000.0 << " ms"
               << std::setw(8) << (sec > 0 ? static_cast<double>(n) / sec / 1e6 : 0.0) << " Mnps";
            if (n != expected) { ss << "   expected " << expected; ok = false; }
            out(ss.str());
        }
    }
 
    // Incremental hash / evaluation / bitboard consistency over whole subtrees.
    bool consistent = true;
    U64 visited = 0;
    for (const PerftCase& c : perftSuite()) {
        if (c.quickDepth == 0) continue;
        pos.setFen(c.fen);
        if (!perftVerify(pos, 3, visited)) { consistent = false; out(std::string("info string state check FAILED: ") + c.name); }
    }
    out(std::string("info string make/unmake consistency (hash, eval, bitboards) over ") + std::to_string(visited) +
        " nodes: " + (consistent ? "OK" : "FAILED"));
    ok = ok && consistent;
 
    std::ostringstream ss;
    ss << "info string self-test " << (ok ? "PASSED" : "FAILED") << ": " << totalNodes << " nodes in "
       << std::fixed << std::setprecision(3) << totalSec << " s ("
       << std::setprecision(1) << (totalSec > 0 ? static_cast<double>(totalNodes) / totalSec / 1e6 : 0.0) << " Mnps)";
    out(ss.str());
    return ok;
}
 
// =============================================================================
// 9. Transposition table
// =============================================================================
enum Bound : int { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };
 
struct TTEntry {                      // 16 bytes
    U64 key;
    std::int16_t score;
    Move move;
    std::uint8_t depth;
    std::uint8_t flags;               // bits 0-1: bound, bits 2-7: search generation ("age")
    int bound() const { return flags & 3; }
    int age() const { return flags >> 2; }
};
static_assert(sizeof(TTEntry) == 16, "TTEntry should be 16 bytes");
 
// Two-way bucketed hash table. A bucket is two adjacent entries; on a miss the
// less valuable one (empty, then old generation, then shallow) is replaced.
class TranspositionTable {
public:
    TranspositionTable() = default;
    TranspositionTable(const TranspositionTable&) = delete;
    TranspositionTable& operator=(const TranspositionTable&) = delete;
    ~TranspositionTable() { std::free(table); }
 
    void resize(std::size_t megabytes) {
        std::lock_guard<std::mutex> lock(mutex);
        std::free(table);
        const std::size_t want = std::max<std::size_t>(megabytes, 1) * 1024 * 1024 / sizeof(TTEntry);
        size = 2;
        while (size * 2 <= want) size *= 2;                 // power of two, so masking works
        table = static_cast<TTEntry*>(std::calloc(size, sizeof(TTEntry)));
        if (!table) { std::cerr << "out of memory for hash table\n"; std::exit(1); }
        mask = size - 1;
        age = 0;
    }
    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        std::memset(table, 0, size * sizeof(TTEntry)); age = 0;
    }
    void newSearch() {
        std::lock_guard<std::mutex> lock(mutex);
        age = (age + 1) & 63;
    }
 
    bool probe(U64 key, TTEntry& out_) const {
        std::lock_guard<std::mutex> lock(mutex);
        const TTEntry* b = &table[(key & mask) & ~static_cast<U64>(1)];
        for (int i = 0; i < 2; ++i)
            if (b[i].key == key && b[i].bound()) { out_ = b[i]; return true; }
        return false;
    }
 
    void store(U64 key, Move m, int score, int depth, int bound) {
        std::lock_guard<std::mutex> lock(mutex);
        TTEntry* b = &table[(key & mask) & ~static_cast<U64>(1)];
        TTEntry* e = nullptr;
        for (int i = 0; i < 2; ++i)
            if (b[i].bound() && b[i].key == key) { e = &b[i]; break; }
        if (e) {
            if (m == NO_MOVE) m = e->move;                  // keep the old best move
            if (bound != BOUND_EXACT && depth + 3 < e->depth && e->age() == age) return;
        } else {
            auto value = [&](const TTEntry& t) { return t.bound() ? t.depth - 8 * ((age - t.age()) & 63) : -1000000; };
            e = (value(b[0]) <= value(b[1])) ? &b[0] : &b[1];
        }
        e->key = key;
        e->move = m;
        e->score = static_cast<std::int16_t>(score);
        e->depth = static_cast<std::uint8_t>(depth);
        e->flags = static_cast<std::uint8_t>(bound | (age << 2));
    }
 
private:
    mutable std::mutex mutex;
    TTEntry* table = nullptr;
    std::size_t size = 0;
    U64 mask = 0;
    int age = 0;
};
 
static TranspositionTable TT;
 
// Mate scores are stored relative to the node, not the root.
inline int scoreToTT(int s, int ply)   { return s >= MATE_IN_MAX ? s + ply : s <= -MATE_IN_MAX ? s - ply : s; }
inline int scoreFromTT(int s, int ply) { return s >= MATE_IN_MAX ? s - ply : s <= -MATE_IN_MAX ? s + ply : s; }
 
// =============================================================================
// 10. Search
// =============================================================================
// Static evaluation from the side-to-move's point of view (centipawns).
// Material and piece-square terms are maintained incrementally by Position, so
// this is O(1): blend middlegame/endgame scores by game phase, add a small
// bishop-pair bonus and a tempo bonus.
inline int evaluate(const Position& pos) {
    const int ph = std::min(pos.phase, Eval::PHASE_TOTAL);
    int score = (pos.mg * ph + pos.eg * (Eval::PHASE_TOTAL - ph)) / Eval::PHASE_TOTAL;
    score += 25 * ((popcount(pos.pieces(WHITE, BISHOP)) >= 2) - (popcount(pos.pieces(BLACK, BISHOP)) >= 2));
    return (pos.stm == WHITE ? score : -score) + 10;
}
 
struct Limits {
    int       depth     = 0;          // 0 = no limit
    U64       nodes     = 0;          // 0 = no limit
    long long movetime  = 0;          // ms, 0 = none
    long long timeLeft[2] = {-1, -1}; // ms per side, -1 = not given
    long long inc[2]      = {0, 0};
    int       movestogo = 0;
    bool      infinite  = false;

    // Snapshot of UCI options for this search; changing an option stops the
    // previous search before these values can be changed.
    int       threads = 1;
    int       moveOverhead = 20;
    int       aspirationWindow = 24;
    int       contempt = 0;
    int       rootSide = WHITE;
    bool      useAspiration = true;
    bool      useNullMove = true;
    bool      useLMR = true;
    bool      useReverseFutility = true;
    bool      useDeltaPruning = true;
    bool      useCheckExtensions = true;
};

struct RootMoveResult {
    Move move = NO_MOVE;
    int score = -INF;
    U64 nodes = 0;
    bool completed = false;
    std::vector<Move> pv;
};
 
static int g_lmr[64][64];             // late-move-reduction table
static void initLmr() {
    for (int d = 1; d < 64; ++d)
        for (int m = 1; m < 64; ++m)
            g_lmr[d][m] = static_cast<int>(0.75 + std::log(static_cast<double>(d)) * std::log(static_cast<double>(m)) / 2.25);
}
 
class Searcher {
public:
    Searcher(const Position& root, const Limits& lim, const std::atomic<bool>& stop,
             const std::chrono::steady_clock::time_point* sharedStart = nullptr)
        : pos(root), limits(lim), stopFlag(stop), sharedStart(sharedStart) {
        std::memset(pv, 0, sizeof pv);
        std::memset(pvLen, 0, sizeof pvLen);
        std::memset(killers, 0, sizeof killers);
        std::memset(history, 0, sizeof history);
    }
 
    // Iterative deepening driver. Prints UCI "info" lines, returns the best move.
    Move run();

    // Search one root move with a full window. Used by the root-parallel path.
    RootMoveResult searchRootMove(Move move, int depth);
 
private:
    using Clock = std::chrono::steady_clock;
 
    Position pos;
    Limits limits;
    const std::atomic<bool>& stopFlag;
    const Clock::time_point* sharedStart = nullptr;
    Clock::time_point t0;
    long long softMs = 0, hardMs = 0;   // 0 = unlimited
    U64 nodes = 0;
    bool stopped = false;
    Move rootBest = NO_MOVE;
 
    Move pv[MAX_PLY + 2][MAX_PLY + 2];  // triangular principal-variation table
    int  pvLen[MAX_PLY + 2];
    Move killers[MAX_PLY + 2][2];
    int  history[2][64][64];            // [side][from][to] quiet-move history
 
    long long elapsedMs() const {
        const Clock::time_point start = sharedStart ? *sharedStart : t0;
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
    }
    void setupTime();
    void checkLimits();
    void printInfo(int depth, int score) const;
 
    int  search(int depth, int alpha, int beta, int ply, bool allowNull);
    int  qsearch(int alpha, int beta, int ply);
    void scoreMoves(const MoveList& ml, int* scores, Move ttMove, int ply) const;
    void updateQuietStats(Move best, const Move* tried, int n, int depth, int ply);
};
 
void Searcher::setupTime() {
    const long long OVERHEAD = std::clamp(limits.moveOverhead, 0, 500); // I/O and scheduling reserve
    softMs = hardMs = 0;
    const int us = pos.stm;
    if (limits.movetime > 0) {
        softMs = hardMs = std::max<long long>(1, limits.movetime - OVERHEAD);
    } else if (!limits.infinite && limits.timeLeft[us] >= 0) {
        const long long left = limits.timeLeft[us], inc = limits.inc[us];
        const int mtg = limits.movestogo > 0 ? std::min(limits.movestogo, 50) : 30;
        const long long alloc = left / mtg + inc * 3 / 4;    // target time for this move
        const long long cap = std::max<long long>(1, left - OVERHEAD);
        hardMs = std::min(std::max<long long>(1, alloc * 3), cap);          // never exceed in mid-iteration
        softMs = std::min(std::max<long long>(1, alloc / 2), hardMs);       // do not start a new depth after this
    }
}
 
void Searcher::checkLimits() {
    if (stopFlag.load(std::memory_order_relaxed)) { stopped = true; return; }
    if (limits.nodes && nodes >= limits.nodes) { stopped = true; return; }
    if (hardMs && elapsedMs() >= hardMs) stopped = true;
}
 
void Searcher::printInfo(int depth, int score) const {
    const long long ms = elapsedMs();
    std::ostringstream os;
    os << "info depth " << depth << " score ";
    if (std::abs(score) >= MATE_IN_MAX)
        os << "mate " << (score > 0 ? (MATE - score + 1) / 2 : -(MATE + score) / 2);
    else
        os << "cp " << score;
    os << " nodes " << nodes << " nps " << (ms > 0 ? static_cast<long long>(nodes) * 1000 / ms : static_cast<long long>(nodes))
       << " time " << ms << " pv";
    for (int i = 0; i < pvLen[0]; ++i) os << ' ' << moveToString(pv[0][i]);
    out(os.str());
}
 
// Move ordering: TT move > captures/queen promotions (MVV-LVA) > killers >
// quiet moves by history score > under-promotions.
void Searcher::scoreMoves(const MoveList& ml, int* scores, Move ttMove, int ply) const {
    const int us = pos.stm;
    for (int i = 0; i < ml.count; ++i) {
        const Move m = ml.moves[i];
        const int from = moveFrom(m), to = moveTo(m);
        int s;
        if (m == ttMove) {
            s = 30000000;
        } else if (isCapture(m) || isPromo(m)) {
            s = 20000000;
            if (isCapture(m)) {
                const int victim = (moveFlags(m) == EP_CAPTURE) ? PAWN : typeOf(pos.board[to]);
                s += victim * 8 + (7 - typeOf(pos.board[from]));
            }
            if (isPromo(m)) s += (promoType(m) == QUEEN) ? 5000 : -30000000;
        } else if (m == killers[ply][0]) {
            s = 10000000;
        } else if (m == killers[ply][1]) {
            s = 9000000;
        } else {
            s = history[us][from][to];
        }
        scores[i] = s;
    }
}
 
// A quiet move caused a beta cutoff: remember it as a killer and reward it in
// the history table (penalise the quiet moves tried before it).
void Searcher::updateQuietStats(Move best, const Move* tried, int n, int depth, int ply) {
    constexpr int MAX_HISTORY = 16384;
    if (killers[ply][0] != best) { killers[ply][1] = killers[ply][0]; killers[ply][0] = best; }
    const int us = pos.stm;
    const int bonus = std::min(depth * depth, 1200);
    auto update = [&](Move m, int b) {
        int& h = history[us][moveFrom(m)][moveTo(m)];
        h += b - h * std::abs(b) / MAX_HISTORY;
    };
    update(best, bonus);
    for (int i = 0; i < n; ++i) update(tried[i], -bonus);
}
 
// Quiescence search: resolve captures (and evasions when in check) so that the
// static evaluation is only trusted in quiet positions.
int Searcher::qsearch(int alpha, int beta, int ply) {
    pvLen[ply] = ply;
    ++nodes;
    if ((nodes & 2047) == 0) checkLimits();
    if (stopped) return 0;
    if (ply >= MAX_PLY - 1) return evaluate(pos);
 
    const bool inCheck = pos.inCheck();
    int best = -INF;
    if (!inCheck) {
        best = evaluate(pos);                       // "stand pat"
        if (best >= beta) return best;
        alpha = std::max(alpha, best);
    }
 
    MoveList ml;
    if (inCheck) generateMoves<false>(pos, ml); else generateMoves<true>(pos, ml);
    int scores[256];
    scoreMoves(ml, scores, NO_MOVE, ply);
 
    int legal = 0;
    for (int i = 0; i < ml.count; ++i) {
        int bi = i;
        for (int j = i + 1; j < ml.count; ++j) if (scores[j] > scores[bi]) bi = j;
        std::swap(ml.moves[i], ml.moves[bi]);
        std::swap(scores[i], scores[bi]);
        const Move m = ml.moves[i];
        if (!pos.isLegal(m)) continue;
        ++legal;
 
        // Delta pruning: even winning the victim cannot lift us to alpha.
        if (limits.useDeltaPruning && !inCheck && !isPromo(m)) {
            const int victim = (moveFlags(m) == EP_CAPTURE) ? PAWN : typeOf(pos.board[moveTo(m)]);
            if (best + Eval::MAT_MG[victim] + 200 <= alpha) continue;
        }
 
        pos.makeMove(m);
        const int score = -qsearch(-beta, -alpha, ply + 1);
        pos.unmakeMove(m);
        if (stopped) return 0;
 
        if (score > best) {
            best = score;
            if (score > alpha) {
                alpha = score;
                if (score >= beta) return score;
            }
        }
    }
    if (inCheck && legal == 0) return -MATE + ply;   // checkmated
    return best;
}
 
// Negamax alpha-beta with principal-variation search (PVS).
int Searcher::search(int depth, int alpha, int beta, int ply, bool allowNull) {
    pvLen[ply] = ply;
    const bool root = (ply == 0);
    const bool pvNode = (beta - alpha) > 1;
 
    if (!root) {
        if (pos.halfmove >= 100 || pos.isRepetition() || pos.insufficientMaterial())
            return (pos.stm == limits.rootSide ? limits.contempt : -limits.contempt); // draw
        if (ply >= MAX_PLY - 1) return evaluate(pos);
        // Mate-distance pruning: no line can be better than mating right now.
        alpha = std::max(alpha, -MATE + ply);
        beta  = std::min(beta, MATE - ply - 1);
        if (alpha >= beta) return alpha;
    }
 
    const bool inCheck = pos.inCheck();
    if (inCheck && limits.useCheckExtensions) ++depth; // optional check extension
    if (depth <= 0) return qsearch(alpha, beta, ply);
 
    ++nodes;
    if ((nodes & 2047) == 0) checkLimits();
    if (stopped) return 0;
 
    // ---- transposition table probe ----------------------------------------
    TTEntry tte{};
    const bool ttHit = TT.probe(pos.key, tte);
    Move ttMove = ttHit ? tte.move : NO_MOVE;
    if (root && rootBest != NO_MOVE) ttMove = rootBest;
    if (ttHit && !pvNode && tte.depth >= depth) {
        const int s = scoreFromTT(tte.score, ply);
        const int b = tte.bound();
        if (b == BOUND_EXACT || (b == BOUND_LOWER && s >= beta) || (b == BOUND_UPPER && s <= alpha)) return s;
    }
 
    const int staticEval = inCheck ? -INF : evaluate(pos);
 
    if (!pvNode && !inCheck && std::abs(beta) < MATE_IN_MAX) {
        // Reverse futility pruning: far above beta at low depth -> fail high.
        if (limits.useReverseFutility && depth <= 6 && staticEval - 90 * depth >= beta)
            return staticEval - 90 * depth;
 
        // Null-move pruning: if passing the turn still fails high, a real move will too.
        if (limits.useNullMove && allowNull && depth >= 3 && staticEval >= beta && pos.hasNonPawnMaterial(pos.stm)) {
            const int R = 3 + depth / 4;
            pos.makeNull();
            const int score = -search(depth - 1 - R, -beta, -beta + 1, ply + 1, false);
            pos.unmakeNull();
            if (stopped) return 0;
            if (score >= beta) return score >= MATE_IN_MAX ? beta : score;
        }
    }
 
    // ---- move loop ----------------------------------------------------------
    MoveList ml;
    generateMoves<false>(pos, ml);
    int scores[256];
    scoreMoves(ml, scores, ttMove, ply);
 
    const int alphaOrig = alpha;
    int bestScore = -INF, legal = 0, nQuiets = 0;
    Move bestMove = NO_MOVE;
    Move quiets[64];
 
    for (int i = 0; i < ml.count; ++i) {
        int bi = i;                                    // selection sort: next-best move first
        for (int j = i + 1; j < ml.count; ++j) if (scores[j] > scores[bi]) bi = j;
        std::swap(ml.moves[i], ml.moves[bi]);
        std::swap(scores[i], scores[bi]);
        const Move m = ml.moves[i];
        if (!pos.isLegal(m)) continue;
        ++legal;
 
        const bool quiet = !isCapture(m) && !isPromo(m);
        pos.makeMove(m);
        const bool givesCheck = pos.inCheck();
        const int newDepth = depth - 1;
        int score;
 
        if (legal == 1) {
            score = -search(newDepth, -beta, -alpha, ply + 1, true);
        } else {
            // Late-move reductions for quiet moves that come late in the ordering.
            int R = 0;
            if (limits.useLMR && depth >= 3 && legal > 3 && quiet && !inCheck && !givesCheck) {
                R = g_lmr[std::min(depth, 63)][std::min(legal, 63)];
                if (pvNode) --R;
                R = std::clamp(R, 0, newDepth - 1);
            }
            score = -search(newDepth - R, -alpha - 1, -alpha, ply + 1, true);     // null window
            if (score > alpha && R > 0) score = -search(newDepth, -alpha - 1, -alpha, ply + 1, true);
            if (score > alpha && score < beta) score = -search(newDepth, -beta, -alpha, ply + 1, true);
        }
        pos.unmakeMove(m);
        if (stopped) return 0;
 
        if (score > bestScore) {
            bestScore = score;
            bestMove = m;
            if (score > alpha) {
                alpha = score;
                pv[ply][ply] = m;                      // splice child PV behind this move
                for (int j = ply + 1; j < pvLen[ply + 1]; ++j) pv[ply][j] = pv[ply + 1][j];
                pvLen[ply] = pvLen[ply + 1];
                if (score >= beta) {
                    if (quiet) updateQuietStats(m, quiets, nQuiets, depth, ply);
                    break;
                }
            }
        }
        if (quiet && nQuiets < 64) quiets[nQuiets++] = m;
    }
 
    if (legal == 0) return inCheck ? -MATE + ply : 0;           // checkmate or stalemate
 
    const int bound = bestScore >= beta ? BOUND_LOWER : (bestScore > alphaOrig ? BOUND_EXACT : BOUND_UPPER);
    TT.store(pos.key, bestMove, scoreToTT(bestScore, ply), depth, bound);
    return bestScore;
}
 
RootMoveResult Searcher::searchRootMove(Move move, int depth) {
    t0 = sharedStart ? *sharedStart : Clock::now();
    setupTime();

    RootMoveResult result;
    result.move = move;
    pos.makeMove(move);
    result.score = -search(depth - 1, -INF, INF, 1, true);
    pos.unmakeMove(move);

    result.nodes = nodes;
    result.completed = !stopped && !stopFlag.load(std::memory_order_relaxed);
    result.pv.push_back(move);
    for (int i = 1; i < pvLen[1] && result.pv.size() < static_cast<std::size_t>(MAX_PLY); ++i)
        result.pv.push_back(pv[1][i]);
    return result;
}

Move Searcher::run() {
    t0 = Clock::now();
    setupTime();
 
    std::vector<Move> legalMoves;
    {
        MoveList all;
        generateMoves<false>(pos, all);
        for (int i = 0; i < all.count; ++i)
            if (pos.isLegal(all.moves[i])) legalMoves.push_back(all.moves[i]);
    }
    if (legalMoves.empty()) {                                  // mate or stalemate on the board
        out(pos.inCheck() ? "info depth 0 score mate 0" : "info depth 0 score cp 0");
        return NO_MOVE;
    }
 
    Move best = legalMoves[0];                                  // fallback if stopped instantly
    int prevScore = 0;
    const int maxDepth = (limits.depth > 0) ? std::min(limits.depth, MAX_PLY - 2) : MAX_PLY - 2;
    const bool parallelRoot = limits.threads > 1 && limits.nodes == 0 && legalMoves.size() > 1;
 
    for (int depth = 1; depth <= maxDepth; ++depth) {
        int score = 0;
        if (parallelRoot) {
            // Each root move gets a full-window search on its own worker. The TT
            // is synchronized, and every helper uses this search's shared clock.
            std::vector<RootMoveResult> results(legalMoves.size());
            std::atomic<std::size_t> next{0};
            const unsigned workerCount = static_cast<unsigned>(
                std::min<std::size_t>(static_cast<std::size_t>(limits.threads), legalMoves.size()));
            std::vector<std::thread> workers;
            workers.reserve(workerCount);
            for (unsigned worker = 0; worker < workerCount; ++worker) {
                workers.emplace_back([&, worker]() {
                    (void)worker;
                    while (!stopFlag.load(std::memory_order_relaxed)) {
                        const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                        if (i >= legalMoves.size()) break;
                        Searcher helper(pos, limits, stopFlag, &t0);
                        results[i] = helper.searchRootMove(legalMoves[i], depth);
                    }
                });
            }
            for (auto& thread : workers) if (thread.joinable()) thread.join();

            bool iterationComplete = !stopFlag.load(std::memory_order_relaxed);
            int bestScore = -INF;
            const RootMoveResult* bestResult = nullptr;
            U64 iterationNodes = 0;
            for (const RootMoveResult& result : results) {
                iterationNodes += result.nodes;
                if (!result.completed) iterationComplete = false;
                if (result.completed && (!bestResult || result.score > bestScore)) {
                    bestScore = result.score;
                    bestResult = &result;
                }
            }
            nodes += iterationNodes;
            if (!iterationComplete || !bestResult) { stopped = true; break; }

            score = bestResult->score;
            pvLen[0] = static_cast<int>(std::min<std::size_t>(bestResult->pv.size(), MAX_PLY));
            for (int i = 0; i < pvLen[0]; ++i) pv[0][i] = bestResult->pv[static_cast<std::size_t>(i)];
        } else {
            // Original single-threaded PVS/aspiration search, kept as the default path.
            int alpha = -INF, beta = INF;
            int delta = std::max(1, limits.aspirationWindow);
            if (limits.useAspiration && depth >= 5) {
                alpha = std::max(prevScore - delta, -INF);
                beta  = std::min(prevScore + delta, INF);
            }
            for (;;) {
                score = search(depth, alpha, beta, 0, false);
                if (stopped) break;
                if (score <= alpha) {
                    beta = (alpha + beta) / 2;
                    alpha = std::max(score - delta, -INF);
                } else if (score >= beta) {
                    beta = std::min(score + delta, INF);
                } else break;
                delta += delta / 2;
            }
            if (stopped) break;                                 // discard the unfinished iteration
        }
        if (stopped) break;
 
        best = pv[0][0];
        rootBest = best;
        prevScore = score;
        printInfo(depth, score);
 
        if (softMs && elapsedMs() >= softMs) break;             // not enough time for another depth
        if (softMs && std::abs(score) >= MATE_IN_MAX) break;    // forced mate found, no need to think on
    }
    return best;
}
 
// =============================================================================
// 11. UCI front-end
// =============================================================================
struct UciOptions {
    int hashMb = 64;
    int threads = 1;
    int moveOverhead = 20;
    int aspirationWindow = 24;
    int contempt = 0;
    bool useAspiration = true;
    bool useNullMove = true;
    bool useLMR = true;
    bool useReverseFutility = true;
    bool useDeltaPruning = true;
    bool useCheckExtensions = true;
};

class UciEngine {
public:
    UciEngine() { pos.setFen(START_FEN); }
    ~UciEngine() { quit(); }
    void loop();
 
private:
    Position pos;
    UciOptions options;
    std::thread worker;
    std::atomic<bool> stopFlag{false};
    bool bounded = false;        // current search ends by itself soon (depth/nodes only)
 
    void stopSearch() {
        stopFlag = true;
        if (worker.joinable()) worker.join();
    }
    // `quit`/EOF: a depth- or node-limited search is allowed to finish (so piped,
    // scripted sessions print their result); anything open-ended is stopped.
    void quit() {
        if (worker.joinable()) {
            if (!bounded) stopFlag = true;
            worker.join();
        }
    }
 
    void uciCommand();
    void positionCommand(std::istringstream& is);
    void goCommand(std::istringstream& is);
    void setOptionCommand(std::istringstream& is);
    void perftCommand(int depth);
};
 
void UciEngine::uciCommand() {
    out("id name Duckie 1.0");
    out("id author rabbbbit-64");
    out("option name Hash type spin default 64 min 1 max 16384");
    out("option name Threads type spin default 1 min 1 max 32");
    out("option name Move Overhead type spin default 20 min 0 max 500");
    out("option name Aspiration Window type spin default 24 min 1 max 200");
    out("option name Contempt type spin default 0 min -100 max 100");
    out("option name Use Aspiration Windows type check default true");
    out("option name Use Null Move Pruning type check default true");
    out("option name Use Late Move Reductions type check default true");
    out("option name Use Reverse Futility Pruning type check default true");
    out("option name Use Delta Pruning type check default true");
    out("option name Use Check Extensions type check default true");
    out("option name Clear Hash type button");
    out("uciok");
}
 
void UciEngine::positionCommand(std::istringstream& is) {
    std::string tok;
    is >> tok;
    Position p;
    if (tok == "startpos") {
        p.setFen(START_FEN);
        is >> tok;
    } else if (tok == "fen") {
        std::string fen;
        while (is >> tok && tok != "moves") fen += tok + ' ';
        if (!p.setFen(fen)) { out("info string invalid FEN"); return; }
    } else {
        return;
    }
    if (tok == "moves") {
        std::string ms;
        while (is >> ms) {
            const Move m = parseMove(p, ms);
            if (m == NO_MOVE) { out("info string illegal move: " + ms); break; }
            p.makeMove(m);
        }
    }
    pos = std::move(p);
}
 
void UciEngine::goCommand(std::istringstream& is) {
    Limits lim;
    std::string tok;
    while (is >> tok) {
        if (tok == "depth") is >> lim.depth;
        else if (tok == "nodes") is >> lim.nodes;
        else if (tok == "movetime") is >> lim.movetime;
        else if (tok == "wtime") is >> lim.timeLeft[WHITE];
        else if (tok == "btime") is >> lim.timeLeft[BLACK];
        else if (tok == "winc") is >> lim.inc[WHITE];
        else if (tok == "binc") is >> lim.inc[BLACK];
        else if (tok == "movestogo") is >> lim.movestogo;
        else if (tok == "infinite") lim.infinite = true;
        else if (tok == "perft") { int d = 1; is >> d; perftCommand(d); return; }
    }
    lim.threads = options.threads;
    lim.moveOverhead = options.moveOverhead;
    lim.aspirationWindow = options.aspirationWindow;
    lim.contempt = options.contempt;
    lim.rootSide = pos.stm;
    lim.useAspiration = options.useAspiration;
    lim.useNullMove = options.useNullMove;
    lim.useLMR = options.useLMR;
    lim.useReverseFutility = options.useReverseFutility;
    lim.useDeltaPruning = options.useDeltaPruning;
    lim.useCheckExtensions = options.useCheckExtensions;
    const bool timed = lim.movetime > 0 || lim.timeLeft[pos.stm] >= 0;
    if (!timed && lim.depth <= 0 && lim.nodes == 0) lim.infinite = true;   // plain `go`
 
    stopSearch();
    stopFlag = false;
    bounded = !lim.infinite && !timed;
    // Draw contempt is root-side-relative, so cached bounds from another root
    // side must not be reused when contempt is enabled.
    if (lim.contempt != 0) TT.clear();
    TT.newSearch();
 
    worker = std::thread([this, root = pos, lim]() {
        auto searcher = std::make_unique<Searcher>(root, lim, stopFlag);
        const Move best = searcher->run();
        if (lim.infinite)                                       // UCI: no bestmove before `stop`
            while (!stopFlag.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        out("bestmove " + moveToString(best));
    });
}
 
void UciEngine::setOptionCommand(std::istringstream& is) {
    std::string tok, name, value;
    is >> tok;                                                  // "name"
    while (is >> tok && tok != "value") name += (name.empty() ? "" : " ") + tok;
    while (is >> tok) value += (value.empty() ? "" : " ") + tok;
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (name == "clear hash") {
        stopSearch();
        TT.clear();
        return;
    }

    auto parseLong = [&]() -> long {
        char* end = nullptr;
        const long parsed = std::strtol(value.c_str(), &end, 10);
        return (end == value.c_str() || *end != '\0') ? 0L : parsed;
    };
    auto parseBool = [&]() -> bool {
        return value == "true" || value == "1" || value == "yes";
    };

    const bool known = name == "hash" || name == "threads" || name == "move overhead" ||
        name == "aspiration window" || name == "contempt" ||
        name == "use aspiration windows" || name == "use null move pruning" ||
        name == "use late move reductions" || name == "use reverse futility pruning" ||
        name == "use delta pruning" || name == "use check extensions";
    if (!known) {
        out("info string unknown option: " + name);
        return;
    }

    stopSearch();
    if (name == "hash") {
        const long mb = std::clamp(parseLong(), 1L, 16384L);
        options.hashMb = static_cast<int>(mb);
        TT.resize(static_cast<std::size_t>(mb));
    } else if (name == "threads") {
        options.threads = static_cast<int>(std::clamp(parseLong(), 1L, 32L));
    } else if (name == "move overhead") {
        options.moveOverhead = static_cast<int>(std::clamp(parseLong(), 0L, 500L));
    } else if (name == "aspiration window") {
        options.aspirationWindow = static_cast<int>(std::clamp(parseLong(), 1L, 200L));
    } else if (name == "contempt") {
        const int newContempt = static_cast<int>(std::clamp(parseLong(), -100L, 100L));
        if (newContempt != options.contempt) TT.clear();
        options.contempt = newContempt;
    } else if (name == "use aspiration windows") {
        options.useAspiration = parseBool();
    } else if (name == "use null move pruning") {
        options.useNullMove = parseBool();
    } else if (name == "use late move reductions") {
        options.useLMR = parseBool();
    } else if (name == "use reverse futility pruning") {
        options.useReverseFutility = parseBool();
    } else if (name == "use delta pruning") {
        options.useDeltaPruning = parseBool();
    } else if (name == "use check extensions") {
        options.useCheckExtensions = parseBool();
    }
}
 
// "divide" output: node count below every legal root move.
void UciEngine::perftCommand(int depth) {
    stopSearch();
    depth = std::max(depth, 1);
    Position p = pos;
    MoveList ml;
    generateMoves<false>(p, ml);
    const auto t0 = std::chrono::steady_clock::now();
    U64 total = 0;
    for (int i = 0; i < ml.count; ++i) {
        const Move m = ml.moves[i];
        if (!p.isLegal(m)) continue;
        p.makeMove(m);
        const U64 n = perft(p, depth - 1);
        p.unmakeMove(m);
        total += n;
        out(moveToString(m) + ": " + std::to_string(n));
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::ostringstream ss;
    ss << "\nNodes searched: " << total << "  (" << std::fixed << std::setprecision(3) << sec << " s, "
       << std::setprecision(1) << (sec > 0 ? static_cast<double>(total) / sec / 1e6 : 0.0) << " Mnps)";
    out(ss.str());
}
 
void UciEngine::loop() {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string cmd;
        if (!(is >> cmd)) continue;
 
        if (cmd == "quit") { quit(); return; }
        else if (cmd == "uci") uciCommand();
        else if (cmd == "isready") out("readyok");
        else if (cmd == "ucinewgame") { stopSearch(); TT.clear(); pos.setFen(START_FEN); }
        else if (cmd == "position") positionCommand(is);
        else if (cmd == "go") goCommand(is);
        else if (cmd == "stop") stopSearch();
        else if (cmd == "setoption") setOptionCommand(is);
        // ---- non-standard helpers ----
        else if (cmd == "perft") { int d = 1; is >> d; perftCommand(d); }
        else if (cmd == "d") out(pos.toString());
        else if (cmd == "eval") out("info string static eval (side to move): " + std::to_string(evaluate(pos)) + " cp");
        else if (cmd == "selftest") { stopSearch(); runSelfTest(false); }
    }
    quit();                                                     // EOF on stdin
}
 
// =============================================================================
// 12. main
// =============================================================================
int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
 
    bool selfTest = true, fullOnly = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--no-selftest") selfTest = false;
        else if (a == "--full") fullOnly = true;
        else if (a == "--help" || a == "-h") {
            std::cout << "Duckie 1.0 - UCI chess engine\n"
                         "  --no-selftest   skip the start-up perft self-test\n"
                         "  --full          run the deep perft suite and exit\n";
            return 0;
        }
    }
 
    Attacks::init();
    Zobrist::init();
    Eval::init();
    initCastleMask();
    initLmr();
    TT.resize(64);
 
    if (fullOnly) return runSelfTest(true) ? 0 : 1;
    if (selfTest && !runSelfTest(false)) {
        out("info string self-test failed - refusing to start");
        return 1;
    }
 
    UciEngine engine;
    engine.loop();
    return 0;
}
 