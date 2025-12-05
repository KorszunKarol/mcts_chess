#!/usr/bin/env python3
"""
Validate Coach Tal by comparing its move choices against the baseline.

This script analyzes several positions and shows:
1. What move the baseline (pure policy) would choose
2. What move Coach Tal chooses
3. The cognitive metrics explaining the difference

This helps verify that Coach Tal is actually making different (hopefully better
against humans) choices in positions where it matters.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import chess
from src.coach_tal import (
    CoachTalSelector, CoachTalConfig, Explainer,
    TransformerEvaluator, entropy, user_ease
)


def analyze_position(name: str, fen: str, selector: CoachTalSelector, 
                     evaluator: TransformerEvaluator, explainer: Explainer):
    """Analyze a single position comparing baseline vs Coach Tal."""
    board = chess.Board(fen)
    
    # Get baseline (pure policy) choice
    _, policy = evaluator.evaluate(board)
    baseline_move = max(policy.items(), key=lambda x: x[1])[0]
    baseline_prob = policy[baseline_move]
    
    # Get Coach Tal choice
    result = selector.select_from_board(board, top_k=5)
    coach_move = result.chosen_move
    
    print(f"\n{'='*60}")
    print(f"Position: {name}")
    print(f"FEN: {fen}")
    print(f"{'='*60}")
    
    # Did Coach Tal choose differently?
    same_move = baseline_move == coach_move
    
    print(f"\nBaseline choice: {board.san(baseline_move)} (prob: {baseline_prob:.1%})")
    print(f"Coach Tal choice: {board.san(coach_move)}")
    print(f"Same move: {'✓ Yes' if same_move else '✗ No - DIFFERENT!'}")
    
    if not same_move:
        # Show why Coach Tal chose differently
        analysis = explainer.explain(result, board)
        print(f"\nCoach Tal's reasoning:")
        print(f"  Move type: {analysis.move_type}")
        print(f"  J-score: {analysis.j_score:.3f}")
        print(f"  Value: {analysis.value:+.3f}")
        print(f"  Opponent entropy: {analysis.opponent_entropy:.2f} nats")
        print(f"  User ease: {analysis.user_ease:.1%}")
        print(f"  Reason: {analysis.primary_reason}")
        
        # Find the baseline move in candidates
        for c in result.all_candidates:
            if c.move == baseline_move:
                print(f"\nBaseline move analysis:")
                print(f"  J-score: {c.j_score:.3f}")
                print(f"  Value: {c.value_after:+.3f}")
                print(f"  Opponent entropy: {c.opponent_entropy:.2f} nats")
                print(f"  Difference: Coach Tal J={analysis.j_score:.3f} vs Baseline J={c.j_score:.3f}")
                break
    
    return same_move


def main():
    print("=" * 60)
    print("Coach Tal Validation - Comparing with Baseline")
    print("=" * 60)
    
    # Initialize
    config = CoachTalConfig(
        weights_path=str(project_root / 'saved_models' / 'best_model_pytorch.pt'),
        use_pytorch=True,
        lambda_psych=0.3,
        gamma_confusion=0.5,
        delta_soundness=0.15,
        top_k_candidates=5,
    )
    
    selector = CoachTalSelector(config)
    evaluator = TransformerEvaluator(
        weights_path=config.weights_path,
        use_pytorch=True,
    )
    explainer = Explainer()
    
    # Test positions - mix of openings and middlegame
    # These are positions where there might be multiple reasonable moves
    positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Italian Game - Black to move", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
        ("Sicilian - White to move", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"),
        ("Caro-Kann - White to move", "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
        ("French Defense - White to move", "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
        ("Queen's Gambit - Black to move", "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2"),
        ("Middlegame - Complex", "r1bq1rk1/ppp2ppp/2n1pn2/3p4/1bPP4/2N1PN2/PP2BPPP/R1BQK2R w KQ - 4 7"),
        ("Middlegame - Tactical", "r2qkb1r/ppp2ppp/2n1bn2/3pp3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5"),
        ("Endgame - Rook ending", "8/5pk1/6p1/8/8/6P1/5PK1/4R3 w - - 0 1"),
        ("Sharp position", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ]
    
    same_count = 0
    diff_count = 0
    
    for name, fen in positions:
        same = analyze_position(name, fen, selector, evaluator, explainer)
        if same:
            same_count += 1
        else:
            diff_count += 1
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Positions analyzed: {len(positions)}")
    print(f"Same move as baseline: {same_count}")
    print(f"Different move (Coach Tal override): {diff_count}")
    print(f"Override rate: {diff_count / len(positions):.1%}")
    
    if diff_count == 0:
        print("\n⚠️  Coach Tal chose the same move in all positions.")
        print("   This might mean:")
        print("   1. The baseline is already optimal for these positions")
        print("   2. lambda_psych is too low to override")
        print("   3. Try more complex/tactical positions")
    else:
        print(f"\n✓ Coach Tal made {diff_count} different choices based on cognitive metrics!")


if __name__ == "__main__":
    main()





