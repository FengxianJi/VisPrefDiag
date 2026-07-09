import os
import time
import gradio as gr
import ujson as json
import traceback
import numpy as np
from typing import List
from tqdm import tqdm
from src.auctioneer_base import Auctioneer
from src.bidder_base import Bidder, bidders_to_chatbots, bidding_multithread
from utils import trace_back

# Load .env file
def load_env_file(env_path='.env'):
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env_file()


LOG_DIR = 'logs'
enable_gr = gr.update(interactive=True)
disable_gr = gr.update(interactive=False)


def monitor_all(bidder_list: List[Bidder]):
    return sum([bidder.to_monitors() for bidder in bidder_list], [])


def parse_bid_price(auctioneer: Auctioneer, bidder: Bidder, msg: str):
    # rebid if the message is not parsible into a bid price
    bid_price = auctioneer.parse_bid(msg)
    while bid_price is None:
        re_msg = bidder.bid("You must be clear about your bidding decision, say either \"I'm out!\" or \"I bid $xxx!\". Please rebid.")
        bid_price = auctioneer.parse_bid(re_msg)
        print(f"{bidder.name} rebid: {re_msg}")
    return bid_price


def enable_human_box(bidder_list):
    signals = []
    for bidder in bidder_list:
        if 'human' in bidder.model_name and not bidder.withdraw:
            signals.append(gr.update(interactive=True, visible=True, 
                                     placeholder="Please bid! Enter \"I'm out\" or \"I bid $xxx\"."))
        else:
            signals.append(disable_gr)
    return signals


def disable_all_box(bidder_list):
    signals = []
    for bidder in bidder_list:
        if 'human' in bidder.model_name: 
            signals.append(gr.update(interactive=False, visible=True, 
                                     placeholder="Wait a moment to engage in the auction."))
        else:
            signals.append(gr.update(interactive=False, visible=False))
    return signals


def run_auction(
    auction_hash: str, 
    auctioneer: Auctioneer, 
    bidder_list: List[Bidder], 
    thread_num: int, 
    yield_for_demo=True,
    log_dir=LOG_DIR,
    repeat_num=0,
    memo_file=None):
    
    # bidder_list[0].verbose=True
    
    if yield_for_demo:
        chatbot_list = bidders_to_chatbots(bidder_list)
        yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [auctioneer.log()] + [disable_gr, disable_gr] + disable_all_box(bidder_list)
    
    # ***************** Learn Round ****************
    for bidder in bidder_list:
        if bidder.enable_learning and memo_file:
            # if no prev memo file, then no need to learn.
            if os.path.exists(memo_file):
                with open(memo_file) as f:
                    data = json.load(f)
                    past_learnings = data['learnings'][bidder.name]
                    past_auction_log = data['auction_log']
                    bidder.learn_from_prev_auction(past_learnings, past_auction_log)
    
    # ***************** Plan Round *****************
    # init bidder profit
    bidder_profit_info = auctioneer.gather_all_status(bidder_list)
    for bidder in bidder_list:
        bidder.set_all_bidders_status(bidder_profit_info)

    plan_instructs = [bidder.get_plan_instruct(auctioneer.items) for bidder in bidder_list]

    bidding_multithread(bidder_list, plan_instructs, func_type='plan', thread_num=thread_num)

    if yield_for_demo:
        chatbot_list = bidders_to_chatbots(bidder_list)
        yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [auctioneer.log()] + [disable_gr, disable_gr] + disable_all_box(bidder_list)

    # ***************** Phase 0: Pre-auction Global Observation *****************
    auctioneer.pre_auction_observation(bidder_list)

    if yield_for_demo:
        chatbot_list = bidders_to_chatbots(bidder_list)
        yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [auctioneer.log()] + [disable_gr, disable_gr] + disable_all_box(bidder_list)

    bar = tqdm(total=len(auctioneer.items_queue), desc='Auction Progress')
    while not auctioneer.end_auction():
        cur_item = auctioneer.present_item()
        
        bid_round = 0
        MAX_BID_ROUNDS = 2
        while True:
            # ***************** Bid Round ***************** 
            auctioneer_msg = auctioneer.ask_for_bid(bid_round)
            _bidder_list = []
            _bid_instruct_list = []
            # remove highest bidder and withdrawn bidders
            for bidder in bidder_list:
                if bidder is auctioneer.highest_bidder or bidder.withdraw:
                    bidder.need_input = False
                    continue
                else:
                    bidder.need_input = True    # enable input from demo
                    instruct = bidder.get_bid_instruct(auctioneer_msg, bid_round)
                    _bidder_list.append(bidder)
                    _bid_instruct_list.append(instruct)
            
            if yield_for_demo:
                chatbot_list = bidders_to_chatbots(bidder_list)
                yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [auctioneer.log()] + [disable_gr, disable_gr] + enable_human_box(bidder_list)
            
            _msgs = bidding_multithread(_bidder_list, _bid_instruct_list, func_type='bid', thread_num=thread_num)

            for i, (msg, bidder) in enumerate(zip(_msgs, _bidder_list)):
                if bidder.model_name == 'rule':
                    bid_price = bidder.bid_rule(auctioneer.prev_round_max_bid, auctioneer.min_markup_pct)
                else:
                    bid_price = parse_bid_price(auctioneer, bidder, msg)

                # Mandatory bidding: force rebid if withdrawal in round 1
                rebid_attempts = 0
                while bid_price is not None and bid_price < 0 and bid_round == 0 and rebid_attempts < 3:
                    rebid_attempts += 1
                    rebid_msg = "You MUST place a bid in the first round. Withdrawing is not allowed. Please bid with \"I bid $xxx!\"."
                    rebid_instruct = bidder.get_rebid_instruct(rebid_msg)
                    msg = bidder.rebid_for_failure(rebid_instruct)
                    bid_price = parse_bid_price(auctioneer, bidder, msg)

                # Fallback: if still refusing in round 1, force starting price
                if bid_price is not None and bid_price < 0 and bid_round == 0:
                    bid_price = auctioneer.cur_item.price
                    print(f"  [FORCE BID] {bidder.name} refused to bid in round 1, forcing starting price ${bid_price}")

                # can't bid more than budget or less than previous highest bid
                sanity_retries = 0
                MAX_SANITY_RETRIES = 3
                while True:
                    fail_msg = bidder.bid_sanity_check(bid_price, auctioneer.prev_round_max_bid, auctioneer.min_markup_pct)
                    if fail_msg is None:
                        break
                    else:
                        sanity_retries += 1
                        if sanity_retries >= MAX_SANITY_RETRIES:
                            # Force withdraw after max retries to prevent infinite loop
                            print(f"  [SANITY LIMIT] {bidder.name} failed {MAX_SANITY_RETRIES} sanity checks, forcing withdraw")
                            bid_price = -1  # force withdraw
                            break
                        bidder.need_input = True   # enable input from demo
                        auctioneer_msg = auctioneer.ask_for_rebid(fail_msg=fail_msg, bid_price=bid_price)
                        rebid_instruct = bidder.get_rebid_instruct(auctioneer_msg)

                        if yield_for_demo:
                            chatbot_list = bidders_to_chatbots(bidder_list)
                            yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [auctioneer.log()] + [disable_gr, disable_gr] + disable_all_box(bidder_list)

                        msg = bidder.rebid_for_failure(rebid_instruct)
                        bid_price = parse_bid_price(auctioneer, bidder, msg)

                    if yield_for_demo:
                        chatbot_list = bidders_to_chatbots(bidder_list)
                        yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [auctioneer.log()] + [disable_gr, disable_gr] + disable_all_box(bidder_list)
                
                bidder.set_withdraw(bid_price)

                auctioneer.record_bid({'bidder': bidder, 'bid': bid_price, 'raw_msg': msg}, bid_round)
            
            if yield_for_demo:
                chatbot_list = bidders_to_chatbots(bidder_list)
                yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [auctioneer.log()] + [disable_gr, disable_gr] + disable_all_box(bidder_list)
            
            is_sold = auctioneer.check_hammer(bid_round)
            bid_round += 1
            if is_sold or bid_round >= MAX_BID_ROUNDS: 
                break
            else:
                if auctioneer.fail_to_sell and auctioneer.enable_discount:
                    for bidder in bidder_list:
                        bidder.set_withdraw(0) # back in the game

        # ***************** Summarize (state update only, no LLM call) *****************
        for bidder in bidder_list:
            if bidder is auctioneer.highest_bidder:
                bidder.win_bid(cur_item, auctioneer.highest_bid)
            else:
                bidder.lose_bid(cur_item)

        # record profit information of all bidders for each bidder
        # (not used in the auction, just for belief tracking evaluation)
        bidder_profit_info = auctioneer.gather_all_status(bidder_list)
        for bidder in bidder_list:
            bidder.set_all_bidders_status(bidder_profit_info)

        auctioneer.hammer_fall()
        bar.update(1)

    total_cost = sum([b.openai_cost for b in bidder_list]) + auctioneer.openai_cost
    bidder_reports = [bidder.profit_report() for bidder in bidder_list]
    
    if yield_for_demo:
        chatbot_list = bidders_to_chatbots(bidder_list, profit_report=True)
        yield [bidder_list] + chatbot_list + monitor_all(bidder_list) + [auctioneer.log(bidder_reports) + f'\n## Total Cost: ${total_cost}'] + [disable_gr, enable_gr] + disable_all_box(bidder_list)
    
    memo = {'auction_log': auctioneer.log(show_model_name=False),
            'memo_text': bidder_reports,
            'profit': {bidder.name: bidder.profit for bidder in bidder_list},
            'total_cost': total_cost,
            'learnings': {bidder.name: bidder.learnings for bidder in bidder_list},
            'model_info': {bidder.name: bidder.model_name for bidder in bidder_list}}
    log_bidders(log_dir, auction_hash, bidder_list, repeat_num, memo)

    # Extract clearing prices BEFORE finish_auction() clears the logs
    clearing_prices = {}
    winner_info = {}
    for item_desc, bids in auctioneer.auction_logs.items():
        # Find the hammer price (last bid entry with 'Hammer' in bid_round)
        hammer_entry = [b for b in bids if 'Hammer' in str(b.get('bid_round', ''))]
        if hammer_entry:
            item_name = item_desc.split(',')[0].strip()
            hammer_bid = hammer_entry[0]['bid']
            bidder = hammer_entry[0]['bidder']

            # 处理流拍情况（无人出价）
            if bidder is None:
                # 流拍，不记录到统计中
                continue

            # Extract numeric price
            if isinstance(hammer_bid, str):
                import re
                price_match = re.search(r'\d+', hammer_bid.replace(',', ''))
                if price_match:
                    price = int(price_match.group())
                    clearing_prices[item_name] = price
                    winner_info[item_name] = bidder.name
            else:
                clearing_prices[item_name] = int(hammer_bid)
                winner_info[item_name] = bidder.name

    auctioneer.finish_auction()

    if not yield_for_demo:
        yield total_cost, clearing_prices, winner_info


def log_bidders(log_dir: str, auction_hash: str, bidder_list: List[Bidder], repeat_num: int, memo: dict):
    for bidder in bidder_list:
        log_file = f"{log_dir}/{auction_hash}/{bidder.name.replace(' ', '')}-{repeat_num}.jsonl"
        if not os.path.exists(log_file):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a') as f:
            log_data = bidder.to_monitors(as_json=True)
            f.write(json.dumps(log_data) + '\n')
    
    with open(f"{log_dir}/{auction_hash}/memo-{repeat_num}.json", 'w') as f:
        f.write(json.dumps(memo) + '\n')


def make_auction_hash():
    return str(int(time.time()))


def run_monte_carlo(
    auction_hash: str,
    items_list: List,
    bidders_config_list: List[dict],
    auctioneer_config: dict,
    num_runs: int,
    thread_num: int,
    shuffle: bool = False,
    random_seed: int = None,
    log_dir: str = LOG_DIR,
):
    """
    Run Monte Carlo simulation (multiple independent auctions).

    Args:
        auction_hash: Unique identifier for this set of runs
        items_list: List of Item objects
        bidders_config_list: List of bidder configuration dicts
        auctioneer_config: Auctioneer configuration dict
        num_runs: Number of Monte Carlo runs
        thread_num: Number of threads for parallel bidding
        shuffle: Whether to shuffle items before each run
        random_seed: Random seed for reproducibility
        log_dir: Directory to save logs

    Returns:
        Dictionary containing Monte Carlo statistics
    """
    import random
    import numpy as np
    from collections import defaultdict

    # Set random seed if specified
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Data structure to collect results
    # clearing_prices[item_name] = [price_run1, price_run2, ...]
    clearing_prices = defaultdict(list)
    winner_distribution = defaultdict(lambda: defaultdict(int))
    all_run_logs = []

    print(f"\n{'='*60}")
    print(f"Starting Monte Carlo Simulation")
    print(f"Number of runs: {num_runs}")
    print(f"Number of items: {len(items_list)}")
    print(f"Number of bidders: {len(bidders_config_list)}")
    print(f"Shuffle items: {shuffle}")
    print(f"Random seed: {random_seed}")
    print(f"{'='*60}\n")

    for run_idx in tqdm(range(num_runs), desc='Monte Carlo Runs'):
        try:
            # Create fresh bidders for this run (independent, no memory from previous runs)
            from src.bidder_base import create_bidders
            bidders = []
            for bidder_config in bidders_config_list:
                bidder = Bidder.create(**bidder_config, auction_hash=auction_hash)
                bidders.append(bidder)

            # Create fresh auctioneer for this run
            auctioneer = Auctioneer(**auctioneer_config)
            auctioneer.init_items(items_list.copy())

            # Shuffle items if requested
            if shuffle:
                auctioneer.shuffle_items()

            # Run single auction (no demo mode, no learning between runs)
            auction_results = list(run_auction(
                auction_hash=auction_hash,
                auctioneer=auctioneer,
                bidder_list=bidders,
                thread_num=thread_num,
                yield_for_demo=False,
                log_dir=log_dir,
                repeat_num=run_idx,
                memo_file=None  # No learning between Monte Carlo runs
            ))

            # Extract results - run_auction yields a single tuple (total_cost, clearing_prices, winner_info)
            if len(auction_results) > 0:
                result = auction_results[0]
                if isinstance(result, tuple) and len(result) == 3:
                    total_cost, run_clearing_prices, run_winner_info = result
                else:
                    # Fallback for old format (just total_cost)
                    total_cost = result
                    run_clearing_prices = {}
                    run_winner_info = {}
            else:
                total_cost = 0.0
                run_clearing_prices = {}
                run_winner_info = {}

            # Collect clearing prices and winners
            for item_name, price in run_clearing_prices.items():
                clearing_prices[item_name].append(price)
                winner = run_winner_info[item_name]
                winner_distribution[item_name][winner] += 1

                # Print run result
                print(f"Run {run_idx+1} | {item_name}: ${price} (Winner: {winner})")

            # Save run log
            all_run_logs.append({
                'run': run_idx,
                'clearing_prices': run_clearing_prices,
                'total_cost': total_cost
            })

        except Exception as e:
            print(f"Error in run {run_idx}: {e}")
            print(trace_back(e))
            continue

    # Compute statistics
    print(f"\n{'='*60}")
    print(f"Monte Carlo Results ({num_runs} runs)")
    print(f"{'='*60}\n")

    statistics = {}
    for item_name, prices in clearing_prices.items():
        if len(prices) == 0:
            continue

        prices_array = np.array(prices)
        stats = {
            'mean': float(np.mean(prices_array)),
            'std': float(np.std(prices_array)),
            'median': float(np.median(prices_array)),
            'min': float(np.min(prices_array)),
            'max': float(np.max(prices_array)),
            'count': len(prices),
            'winner_distribution': dict(winner_distribution[item_name])
        }
        statistics[item_name] = stats

        # Print statistics
        print(f"{item_name}:")
        print(f"  Mean Clearing Price: ${stats['mean']:.2f}")
        print(f"  Std Dev: ${stats['std']:.2f}")
        print(f"  Median: ${stats['median']:.2f}")
        print(f"  Range: [${stats['min']:.0f}, ${stats['max']:.0f}]")
        print(f"  Winner Distribution:")
        for winner, count in stats['winner_distribution'].items():
            pct = count / stats['count'] * 100
            print(f"    {winner}: {count} ({pct:.1f}%)")
        print()

    # Save statistics to file
    stats_file = os.path.join(log_dir, auction_hash, 'monte_carlo_stats.json')
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump({
            'config': {
                'num_runs': num_runs,
                'num_items': len(items_list),
                'num_bidders': len(bidders_config_list),
                'shuffle': shuffle,
                'random_seed': random_seed,
            },
            'statistics': statistics,
            'all_runs': all_run_logs
        }, f, indent=2)

    print(f"Statistics saved to: {stats_file}")

    return statistics



if __name__ == '__main__':
    import argparse
    from src.item_base import create_items
    from src.bidder_base import create_bidders
    from transformers import GPT2TokenizerFast
    from config import AuctionConfig as cfg
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    # import cjjpy as cjj

    parser = argparse.ArgumentParser(description='Run auction simulations')
    parser.add_argument('--input_dir', '-i', type=str, default='data/example/', help='Directory containing items and bidders files')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle items before each auction')
    parser.add_argument('--repeat', type=int, default=1, help='Number of repeated auctions (legacy mode, not Monte Carlo)')
    parser.add_argument('--threads', '-t', type=int, help='Number of threads. Max is number of bidders. Reduce if rate limit is low (e.g., GPT-4).', required=True)
    parser.add_argument('--memo_file', '-m', type=str, help='The last memo.json file for learning (only for legacy mode)')

    # Monte Carlo mode arguments
    parser.add_argument('--monte_carlo', action='store_true', help='Run in Monte Carlo mode (multiple independent runs)')
    parser.add_argument('--num_runs', type=int, default=None, help='Number of Monte Carlo runs (default: from config)')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed for reproducibility (default: from config)')

    # Configuration file paths
    parser.add_argument('--items_file', type=str, default=None, help='Items file name (default: items_demo.jsonl)')
    parser.add_argument('--bidders_file', type=str, default=None, help='Bidders file name (default: bidders_demo.jsonl)')

    args = parser.parse_args()

    # Determine file names
    items_file_name = args.items_file if args.items_file else 'items_demo.jsonl'
    bidders_file_name = args.bidders_file if args.bidders_file else 'bidders_demo.jsonl'

    # Load items and bidders configuration
    item_file = os.path.join(args.input_dir, items_file_name)
    bidder_file = os.path.join(args.input_dir, bidders_file_name)

    if not os.path.exists(item_file):
        print(f"Error: Items file not found: {item_file}")
        sys.exit(1)
    if not os.path.exists(bidder_file):
        print(f"Error: Bidders file not found: {bidder_file}")
        sys.exit(1)

    items = create_items(item_file)

    # Load bidder configurations (as dicts for Monte Carlo mode)
    import ujson as json
    bidders_config = []
    with open(bidder_file, 'r') as f:
        for line in f:
            if line.strip():
                bidders_config.append(json.loads(line))

    auction_hash = make_auction_hash()

    # ================ Monte Carlo Mode ================
    if args.monte_carlo:
        print("\n" + "="*60)
        print("Running in Monte Carlo Mode")
        print("="*60)

        # Use config values or command line overrides
        num_runs = args.num_runs if args.num_runs is not None else cfg.NUM_RUNS
        random_seed = args.random_seed if args.random_seed is not None else cfg.RANDOM_SEED
        shuffle = args.shuffle if args.shuffle else cfg.SHUFFLE_ITEMS

        # Check if items are in image mode
        is_image_mode = len(items) > 0 and items[0].is_image_mode

        # Generate contact sheet if in image mode
        if is_image_mode and cfg.USE_CONTACT_SHEET:
            auctioneer_temp = Auctioneer()
            auctioneer_temp.init_items(items)
            contact_sheet_path = auctioneer_temp.generate_contact_sheet()
            if contact_sheet_path:
                print(f"✓ Contact sheet generated: {contact_sheet_path}")

        # Auctioneer configuration
        auctioneer_config = {
            'enable_discount': False,
            'min_markup_pct': 0.1,
        }

        # Set unified budget for all bidders if enabled
        if cfg.UNIFIED_BUDGET and is_image_mode:
            unified_budget = cfg.get_unified_budget(items)
            print(f"✓ Unified budget enabled: ${unified_budget:.2f}")
            for bidder_config in bidders_config:
                bidder_config['budget'] = int(unified_budget)

        # Run Monte Carlo simulation
        statistics = run_monte_carlo(
            auction_hash=auction_hash,
            items_list=items,
            bidders_config_list=bidders_config,
            auctioneer_config=auctioneer_config,
            num_runs=num_runs,
            thread_num=min(args.threads, len(bidders_config)),
            shuffle=shuffle,
            random_seed=random_seed,
            log_dir=args.input_dir,
        )

        print("\n" + "="*60)
        print(f"Monte Carlo simulation completed!")
        print(f"Results saved in: {args.input_dir}/{auction_hash}/")
        print("="*60)

    # ================ Legacy Mode (Original) ================
    else:
        print("\n" + "="*60)
        print("Running in Legacy Mode (Original)")
        print("="*60)

        total_money_spent = 0
        for i in tqdm(range(args.repeat), desc='Repeat'):
            cnt = 3
            while cnt > 0:
                try:
                    memo_file = args.memo_file if args.memo_file else f'{args.input_dir}/{auction_hash}/memo-{i-1}.json'
                    items = create_items(item_file)
                    bidders = create_bidders(bidder_file, auction_hash=auction_hash)

                    # Set unified budget if enabled and in image mode
                    is_image_mode = len(items) > 0 and items[0].is_image_mode
                    if cfg.UNIFIED_BUDGET and is_image_mode:
                        unified_budget = cfg.get_unified_budget(items)
                        print(f"✓ Setting unified budget: ${unified_budget:.2f}")
                        for bidder in bidders:
                            bidder.set_unified_budget(unified_budget)

                    auctioneer = Auctioneer(enable_discount=False)
                    auctioneer.init_items(items)
                    if args.shuffle:
                        auctioneer.shuffle_items()
                    money_spent = list(run_auction(
                        auction_hash,
                        auctioneer,
                        bidders,
                        thread_num=min(args.threads, len(bidders)),
                        yield_for_demo=False,
                        log_dir=args.input_dir,
                        repeat_num=i,
                        memo_file=memo_file,
                    ))
                    total_money_spent += sum(money_spent)
                    break
                except Exception as e:
                    cnt -= 1
                    print(f"Error in {i}th auction: {e}\n{trace_back(e)}")
                    print(f"Retry {cnt} more times...")

        print('Total money spent: $', total_money_spent)
        # cjj.SendEmail(f'Completed: {args.input_dir} - {auction_hash}', f'Total money spent: ${total_money_spent}')

