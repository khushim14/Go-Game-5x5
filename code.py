#!/usr/bin/env python3
import numpy as np
import random
import time
import hashlib

# Global variables
BOARD_SIZE = 5
EMPTY = 0
BLACK = 1
WHITE = 2
KOMI = 2.5

class GoGame:
    def __init__(self, player_color):
        self.board_size = BOARD_SIZE
        self.player_color = player_color
        self.opponent_color = WHITE if player_color == BLACK else BLACK
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.previous_board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.max_depth = 4  # Increased maximum depth
        self.transposition_table = {}  # Cache for previously evaluated positions
        self.move_history = []  # Track history of moves for better ordering
        
    def read_input(self, filename="input.txt"):
        """Read the board from input.txt"""
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        # First line is the player color
        self.player_color = int(lines[0])
        self.opponent_color = WHITE if self.player_color == BLACK else BLACK
        
        # Read previous board state (lines 1-5)
        for i in range(BOARD_SIZE):
            line = lines[i + 1].strip()
            for j in range(BOARD_SIZE):
                self.previous_board[i][j] = int(line[j])
        
        # Read current board state (lines 6-10)
        for i in range(BOARD_SIZE):
            line = lines[i + 5 + 1].strip()
            for j in range(BOARD_SIZE):
                self.board[i][j] = int(line[j])
        
        # Infer move history by comparing boards
        if not np.array_equal(self.previous_board, np.zeros_like(self.previous_board)):
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if self.board[i][j] != self.previous_board[i][j] and self.board[i][j] != EMPTY:
                        self.move_history.append((i, j))
    
    def write_output(self, move, filename="output.txt"):
        """Write the move to output.txt"""
        if move == "PASS":
            output = "PASS"
        else:
            i, j = move
            output = f"{i},{j}"
            
        with open(filename, 'w') as f:
            f.write(output)
    
    def get_valid_moves(self, board, player_color):
        """Get all valid moves for a player on the given board"""
        valid_moves = []
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == EMPTY:
                    # Check if move is valid (not suicidal and doesn't violate KO rule)
                    if self.is_valid_move(board, (i, j), player_color):
                        valid_moves.append((i, j))
        
        # Always include PASS as a valid move
        valid_moves.append("PASS")
        return valid_moves
    
    def is_valid_move(self, board, move, color):
        """Check if a move is valid (not suicidal and doesn't violate KO rule)"""
        if move == "PASS":
            return True
            
        i, j = move
        
        # Check if position is already occupied
        if board[i][j] != EMPTY:
            return False
        
        # Create a board copy to simulate the move
        board_copy = np.copy(board)
        board_copy[i][j] = color
        
        # Check if the move captures any opponent stones
        captured = self.find_captured_stones(board_copy, color)
        if captured:
            # If captures opponent stones, remove them from board copy
            for pos in captured:
                board_copy[pos[0]][pos[1]] = EMPTY
                
        # Check for suicide move
        if not self.has_liberty(board_copy, i, j):
            group = self.find_connected_group(board_copy, i, j)
            if not any(self.has_liberty(board_copy, x, y) for x, y in group):
                return False
        
        # Check for KO rule violation
        if np.array_equal(board_copy, self.previous_board):
            return False
        
        return True
    
    def has_liberty(self, board, i, j):
        """Check if a stone has liberty (empty adjacent position)"""
        color = board[i][j]
        if color == EMPTY:
            return True
            
        for ni, nj in self.get_neighbors(i, j):
            if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE and board[ni][nj] == EMPTY:
                return True
        
        return False
    
    def get_neighbors(self, i, j):
        """Get the neighboring positions"""
        return [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    
    def find_connected_group(self, board, i, j):
        """Find all stones connected to the stone at (i, j)"""
        color = board[i][j]
        if color == EMPTY:
            return []
            
        visited = set()
        stack = [(i, j)]
        
        while stack:
            pos = stack.pop()
            if pos in visited:
                continue
                
            visited.add(pos)
            x, y = pos
            
            for nx, ny in self.get_neighbors(x, y):
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[nx][ny] == color and (nx, ny) not in visited:
                    stack.append((nx, ny))
        
        return visited
    
    def find_captured_stones(self, board, player_color):
        """Find all opponent stones that would be captured by player's move"""
        opponent_color = WHITE if player_color == BLACK else BLACK
        captured = []
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == opponent_color:
                    group = self.find_connected_group(board, i, j)
                    if not any(self.has_liberty(board, x, y) for x, y in group):
                        captured.extend(group)
        
        return captured
    
    def apply_move(self, board, move, color):
        """Apply a move to the board and return the new board state"""
        if move == "PASS":
            return np.copy(board)
            
        board_copy = np.copy(board)
        i, j = move
        board_copy[i][j] = color
        
        # Capture opponent stones if any
        opponent_color = WHITE if color == BLACK else BLACK
        opponent_captured = []
        
        for ni, nj in self.get_neighbors(i, j):
            if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE and board_copy[ni][nj] == opponent_color:
                group = self.find_connected_group(board_copy, ni, nj)
                if not any(self.has_liberty(board_copy, x, y) for x, y in group):
                    opponent_captured.extend(group)
        
        # Remove captured stones
        for x, y in opponent_captured:
            board_copy[x][y] = EMPTY
        
        return board_copy
    
    def calculate_territory(self, board):
        """Estimate territory control using flood fill"""
        territory = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        visited = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == EMPTY and not visited[i][j]:
                    region = []
                    bordering_black = False
                    bordering_white = False
                    
                    # Flood fill
                    stack = [(i, j)]
                    while stack:
                        x, y = stack.pop()
                        if visited[x][y]:
                            continue
                        
                        visited[x][y] = True
                        if board[x][y] == EMPTY:
                            region.append((x, y))
                            
                            for nx, ny in self.get_neighbors(x, y):
                                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                                    if board[nx][ny] == EMPTY and not visited[nx][ny]:
                                        stack.append((nx, ny))
                                    elif board[nx][ny] == BLACK:
                                        bordering_black = True
                                    elif board[nx][ny] == WHITE:
                                        bordering_white = True
                    
                    # Assign territory
                    territory_owner = EMPTY
                    if bordering_black and not bordering_white:
                        territory_owner = BLACK
                    elif bordering_white and not bordering_black:
                        territory_owner = WHITE
                    
                    for x, y in region:
                        territory[x][y] = territory_owner
        
        return territory
    
    def evaluate_board(self, board, player_color):
        """Evaluate the board state for a player with improved heuristics"""
        opponent_color = WHITE if player_color == BLACK else BLACK
        
        # Count stones
        player_stones = np.sum(board == player_color)
        opponent_stones = np.sum(board == opponent_color)
        
        # Count liberties
        player_liberties = 0
        opponent_liberties = 0
        player_groups = []
        opponent_groups = []
        
        visited = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if not visited[i][j]:
                    if board[i][j] == player_color:
                        group = self.find_connected_group(board, i, j)
                        player_groups.append(group)
                        group_liberties = 0
                        
                        for x, y in group:
                            visited[x][y] = True
                            for nx, ny in self.get_neighbors(x, y):
                                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[nx][ny] == EMPTY:
                                    group_liberties += 1
                        
                        player_liberties += group_liberties
                    
                    elif board[i][j] == opponent_color:
                        group = self.find_connected_group(board, i, j)
                        opponent_groups.append(group)
                        group_liberties = 0
                        
                        for x, y in group:
                            visited[x][y] = True
                            for nx, ny in self.get_neighbors(x, y):
                                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[nx][ny] == EMPTY:
                                    group_liberties += 1
                        
                        opponent_liberties += group_liberties
        
        # Territory estimation
        territory = self.calculate_territory(board)
        player_territory = np.sum(territory == player_color)
        opponent_territory = np.sum(territory == opponent_color)
        
        # Control of center and edges
        center_weight = 2.0
        edge_weight = 1.5
        corner_weight = 2.5
        
        player_center = 0
        opponent_center = 0
        player_edge = 0
        opponent_edge = 0
        player_corner = 0
        opponent_corner = 0
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                # Center control (distance from center)
                distance_to_center = abs(i - BOARD_SIZE//2) + abs(j - BOARD_SIZE//2)
                center_value = (BOARD_SIZE - distance_to_center) / BOARD_SIZE
                
                is_corner = (i == 0 or i == BOARD_SIZE-1) and (j == 0 or j == BOARD_SIZE-1)
                is_edge = not is_corner and (i == 0 or i == BOARD_SIZE-1 or j == 0 or j == BOARD_SIZE-1)
                
                if board[i][j] == player_color:
                    if distance_to_center <= 1:
                        player_center += center_weight
                    if is_edge:
                        player_edge += edge_weight
                    if is_corner:
                        player_corner += corner_weight
                elif board[i][j] == opponent_color:
                    if distance_to_center <= 1:
                        opponent_center += center_weight
                    if is_edge:
                        opponent_edge += edge_weight
                    if is_corner:
                        opponent_corner += corner_weight
        
        # Connectivity and group count (fewer larger groups is better than many small groups)
        group_count_weight = 1.5
        player_group_penalty = len(player_groups) * group_count_weight
        opponent_group_penalty = len(opponent_groups) * group_count_weight
        
        # Calculate final score with improved weights
        stone_diff = player_stones - opponent_stones
        liberty_diff = player_liberties - opponent_liberties
        territory_diff = player_territory - opponent_territory
        position_diff = (player_center + player_edge + player_corner) - (opponent_center + opponent_edge + opponent_corner)
        group_diff = opponent_group_penalty - player_group_penalty  # Fewer groups is better
        
        # Weights for different factors
        stone_weight = 2.0
        liberty_weight = 2.5  # Increased from 1.5
        territory_weight = 3.0  # New factor
        position_weight = 1.5  # Increased from 1.0
        
        score = (stone_weight * stone_diff) + \
                (liberty_weight * liberty_diff) + \
                (territory_weight * territory_diff) + \
                (position_weight * position_diff) + \
                group_diff
        
        # Apply Komi if player is white
        if player_color == WHITE:
            score += KOMI
            
        return score
    
    def get_board_hash(self, board):
        """Generate a hash for the board position"""
        return hashlib.md5(board.tobytes()).hexdigest()
    
    def order_moves(self, board, valid_moves, color):
        """Order moves based on heuristics for better alpha-beta pruning"""
        if not valid_moves or len(valid_moves) <= 1:
            return valid_moves
            
        move_scores = []
        
        for move in valid_moves:
            if move == "PASS":
                move_scores.append((move, -100))  # Low priority for PASS
                continue
                
            i, j = move
            score = 0
            
            # Historical move preference from transposition table
            board_hash = self.get_board_hash(board)
            if board_hash in self.transposition_table:
                _, best_move = self.transposition_table[board_hash]
                if best_move == move:
                    score += 50  # Huge bonus for previously good moves
            
            # Prefer moves that were played recently
            if move in self.move_history:
                score += 10
            
            # Prefer capturing moves
            test_board = self.apply_move(board, move, color)
            capture_count = np.sum(board != EMPTY) - np.sum(test_board != EMPTY) + 1  # +1 for the stone just placed
            score += capture_count * 15
            
            # Prefer moves with more liberties
            liberties = 0
            for ni, nj in self.get_neighbors(i, j):
                if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE and test_board[ni][nj] == EMPTY:
                    liberties += 1
            score += liberties * 3
            
            # Prefer strategic positions
            # Corners
            if (i == 0 or i == BOARD_SIZE-1) and (j == 0 or j == BOARD_SIZE-1):
                score += 8
            # Edges
            elif i == 0 or i == BOARD_SIZE-1 or j == 0 or j == BOARD_SIZE-1:
                score += 4
            # Center area
            elif abs(i - BOARD_SIZE//2) <= 1 and abs(j - BOARD_SIZE//2) <= 1:
                score += 6
            
            # Moves adjacent to existing friendly stones
            for ni, nj in self.get_neighbors(i, j):
                if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                    if board[ni][nj] == color:
                        score += 2
            
            move_scores.append((move, score))
        
        # Sort moves by score in descending order
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in move_scores]
    
    def minimax(self, board, depth, alpha, beta, maximizing_player, player_color, current_depth=0):
        """Minimax algorithm with alpha-beta pruning and transposition table"""
        board_hash = self.get_board_hash(board)
        
        # Check transposition table
        if board_hash in self.transposition_table:
            cached_depth, cached_move = self.transposition_table[board_hash]
            if cached_depth >= depth:
                # We can use the cached result
                return self.evaluate_board(board, self.player_color), cached_move
        
        if depth == 0:
            return self.evaluate_board(board, self.player_color), None
        
        current_color = self.player_color if maximizing_player else self.opponent_color
        
        # Get valid moves and sort them for better pruning
        valid_moves = self.get_valid_moves(board, current_color)
        if not valid_moves:
            return self.evaluate_board(board, self.player_color), None
            
        # Order moves to improve alpha-beta pruning
        ordered_moves = self.order_moves(board, valid_moves, current_color)
        
        best_value = float('-inf') if maximizing_player else float('inf')
        best_move = None
        
        for move in ordered_moves:
            new_board = self.apply_move(board, move, current_color)
            
            # Check if this is a quiescent position (captures or threatens captures)
            is_quiet = True
            if move != "PASS":
                # If we just captured stones or placed a stone with few liberties
                new_hash = self.get_board_hash(new_board)
                if np.sum(board != EMPTY) - np.sum(new_board != EMPTY) + 1 > 1:  # Captured something
                    is_quiet = False
                
                # Extend search on critical positions
                if not is_quiet and depth == 1 and current_depth < 3:  # Max quiescence depth
                    value, _ = self.minimax(new_board, 1, alpha, beta, not maximizing_player, player_color, current_depth + 1)
                else:
                    value, _ = self.minimax(new_board, depth - 1, alpha, beta, not maximizing_player, player_color, current_depth)
            else:
                value, _ = self.minimax(new_board, depth - 1, alpha, beta, not maximizing_player, player_color, current_depth)
            
            if maximizing_player:
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)
            
            # Alpha-beta pruning
            if beta <= alpha:
                break
        
        # Store result in transposition table
        self.transposition_table[board_hash] = (depth, best_move)
        
        return best_value, best_move
    
    def find_best_move(self):
        """Find the best move using optimized minimax with iterative deepening"""
        start_time = time.time()
        
        # First check the opening move for black
        if np.sum(self.board) == 0 and self.player_color == BLACK:
            return (BOARD_SIZE//2, BOARD_SIZE//2)  # Start in the center
        
        # Check if the board is nearly full, then consider passing
        empty_count = np.sum(self.board == EMPTY)
        if empty_count <= 2:
            player_stones = np.sum(self.board == self.player_color)
            opponent_stones = np.sum(self.board == self.opponent_color)
            
            if self.player_color == BLACK and player_stones > opponent_stones:
                return "PASS"
            elif self.player_color == WHITE and (player_stones + KOMI) > opponent_stones:
                return "PASS"
        
        # Iterative deepening
        best_move = None
        best_move_value = float('-inf')
        
        # Start with depth 1 and increase until time runs out
        for depth in range(1, self.max_depth + 1):
            if time.time() - start_time > 8.0:  # Time limit (9s - 1s buffer)
                break
                
            value, move = self.minimax(self.board, depth, float('-inf'), float('inf'), True, self.player_color)
            
            # Always keep the best move from the latest completed depth
            best_move = move
            best_move_value = value
            
            # Early termination if we find a clearly winning move
            if best_move_value > 100:  # A very high evaluation score
                break
        
        if best_move is None:
            # Fallback: choose a random valid move
            valid_moves = self.get_valid_moves(self.board, self.player_color)
            best_move = random.choice(valid_moves)
        
        return best_move

def main():
    # Create a Go game instance
    game = GoGame(BLACK)  # Default to BLACK, will be updated by reading input
    
    # Read the board state from input.txt
    game.read_input()
    
    # Find the best move
    best_move = game.find_best_move()
    
    # Write the move to output.txt
    game.write_output(best_move)

if __name__ == "__main__":
    main()
