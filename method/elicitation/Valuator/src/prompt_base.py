# for bidder
SYSTEM_MESSAGE = """
You are {name}, who is attending an ascending-bid auction as a bidder. This auction will have some other bidders to compete with you in bidding wars. The price is gradually raised, bidders drop out until finally only one bidder remains, and that bidder wins the item at this final price. Remember: {desire_desc}.

Here are some must-know rules for this auction:

1. Item Evaluation: You need to evaluate items based on the information provided (images, descriptions, context). Your evaluation reflects your subjective preference and aesthetic judgment. Different bidders may have different valuations for the same item.
2. Winning Bid: The highest bid wins the item. You should try to win items you value highly while managing your budget wisely for remaining items.
3. Budget Constraint: You have a fixed budget. Every bid you win consumes your budget, reducing your options for future items. Strategic allocation is crucial.
4. Bidding Rounds: Each item has exactly 1 round of bidding. You must make your best bid in this single round. The highest bidder wins the item.
5. Mandatory Bidding: You MUST place a bid on every item. Withdrawing without bidding is NOT allowed.
""".strip()


_LEARNING_STATEMENT = " and your learnings from previous auctions"


INSTRUCT_PLAN_TEMPLATE = """
As {bidder_name}, you have a total budget of ${budget}. This auction has a total of {item_num} items to be sequentially presented, they are:
{items_info}

---

Please plan for your bidding strategy for the auction based on the information{learning_statement}. A well-thought-out plan positions you advantageously against competitors, allowing you to allocate resources effectively. With a clear strategy, you can make decisions rapidly and confidently, especially under the pressure of the auction environment. Remember: {desire_desc}.

**Important Evaluation Guidelines:**
- Base your judgment ONLY on the aesthetic qualities, visual appeal, and context-fit of each item
- DO NOT infer value from starting prices - they are arbitrary and do not reflect true worth
- Different items may appeal to you differently based on your personal aesthetic preferences
- Consider how well each item fits its stated evaluation context

**CRITICAL — Extreme Differentiation & Penalty Rule:**
- You MUST assign DIFFERENT priority levels to each item. No two items may share the same priority.
- Even if items appear similar, you MUST find distinguishing features (composition, color harmony, detail quality, context-fit, emotional impact, etc.) and rank one strictly above the other.
- Budget allocation MUST be extreme and punitive:
  * Your BEST item: allocate 90-95% of your total budget. Go all-in on the item you truly value.
  * Your WORST item: allocate only 2-5% of your total budget (just a few dollars). Treat it as nearly worthless — you are penalizing it for being inferior.
- Think of this as a penalty system: inferior items deserve almost no investment. The gap between your best and worst allocation should be as large as possible.

After articulating your thinking, assign a priority level to each item. Present the priorities for all items in a JSON format, each item should be represented as a key-value pair, where the key is the item name and the value is its priority on the scale from 1-3. An example output is: {{"Fixture Y": 3, "Module B": 1}}. The descriptions of the priority scale of items are as follows.
    * 1 - This item is nearly worthless to you. Allocate the bare minimum budget ($2-$5). You are penalizing it.
    * 2 - This item has some value but is clearly inferior to the best one. Allocate a small budget.
    * 3 - This item is outstanding and your top priority. Allocate almost your entire budget.

Then, output your budget allocation as a JSON object, for example: {{"Fixture Y": 220, "Module B": 5}}.
""".strip()


INSTRUCT_BID_TEMPLATE = """
Now, the auctioneer says: "{auctioneer_msg}"

**This is round {bid_round}/2 for this item.** After 2 rounds, the highest bidder wins.

---

**Your current budget status: ${remaining_budget} remaining.**
{budget_allocation_instruction}

As {bidder_name}, you have to decide how much to bid on this item according to your plan{learning_statement}. Remember, {desire_desc}.

**MANDATORY BIDDING RULES:**
1. You MUST bid exactly your pre-allocated amount for this item. This is NON-NEGOTIABLE.
2. **HARD CEILING: You are FORBIDDEN from bidding more than your allocated budget for this item, even if you have more remaining budget.** Spending above your allocation wastes money meant for better items.
3. **HARD FLOOR: You MUST bid at least your allocated amount.** Do not bid lower — your plan was made for a reason.
4. The allocation reflects your aesthetic judgment. Trust your plan and execute it precisely.

Give a brief reason, then state your bid clearly: "I bid $xxx!" where xxx equals your allocated amount for this item.
""".strip()


INSTRUCT_SUMMARIZE_TEMPLATE = """
Here is the history of the bidding war of {cur_item}:
"{bidding_history}"

The auctioneer concludes: "{hammer_msg}"

---

{win_lose_msg}
As {bidder_name}, you have to update the status of the auction based on this round of bidding. Here's your previous status:
```
{prev_status}
```

Summarize the notable behaviors of all bidders in this round of bidding for future reference. Then, update the status JSON regarding the following information:
- 'remaining_budget': The remaining budget of you, expressed as a numerical value.
- 'winning_bids': The winning bids for every item won by each bidder, listed as key-value pairs, for example, {{"bidder_name": {{"item_name_1": winning_bid}}, {{"item_name_2": winning_bid}}, ...}}. If a bidder hasn't won any item, then the value for this bidder should be an empty dictionary {{}}.
- Only include the bidders mentioned in the given text. If a bidder is not mentioned (e.g. Bidder 4 in the following example), then do not include it in the JSON object.

After summarizing the bidding history, you must output the current status in a parsible JSON format. An example output looks like:
```
{{"remaining_budget": 8000, "winning_bids": {{"Bidder 1": {{"Item 2": 1200, "Item 3": 1000}}, "Bidder 2": {{"Item 1": 2000}}, "Bidder 3": {{}}}}}}
```
""".strip()


INSTRUCT_LEARNING_TEMPLATE = """
Review and reflect on the historical data provided from a past auction.

{past_auction_log}

Here are your past learnings:

{past_learnings}

Based on the auction log, formulate or update your learning points that could be advantageous to your strategies in the future. Focus on:
1. How well your bidding decisions aligned with your aesthetic preferences
2. Whether you effectively allocated your budget across items
3. Patterns in other bidders' behavior and competition intensity
4. Strategic lessons about when to bid aggressively vs. conservatively

Your learnings should be strategic, of universal relevance, and practical use for future auctions. Consolidate your learnings into a concise numbered list of sentences.
""".strip()


INSTRUCT_REPLAN_TEMPLATE = """
The current status of you and other bidders is as follows:
```
{status_quo}
```

Here are the remaining items in the rest of the auction:
"{remaining_items_info}"

As {bidder_name}, considering the current status{learning_statement}, review your strategies. Adjust your plans based on the outcomes and new information to achieve your primary objective. This iterative process ensures that your approach remains relevant and effective. Please do the following:
1. Always remember: {desire_desc}.
2. Determine and explain if there's a need to update the priority list of remaining items based on the current status, your aesthetic preferences, and your remaining budget.
3. **You MUST assign DIFFERENT priority levels to remaining items.** No two items may share the same priority. Find clear distinguishing features to justify your ranking.
4. Present the updated priorities in a JSON format, each item should be represented as a key-value pair, where the key is the item name and the value is its priority on the scale from 1-3. An example output is: {{"Fixture Y": 3, "Module B": 1}}. The descriptions of the priority scale of items are as follows.
    * 1 - This item has lower aesthetic appeal to you. Consider giving it up if necessary to save budget for more preferred items.
    * 2 - This item has moderate aesthetic appeal but isn't a top priority. Could bid on it if you have enough budget.
    * 3 - This item has high aesthetic appeal and is a top priority for you in the rest of the auction.
""".strip()


# for auctioneer
PARSE_BID_INSTRUCTION = """
Your task is to parse a response from a bidder in an auction, and extract the bidding price from the response. Here are the rules:
- If the language model decides to withdraw from the bidding (e.g., saying "I'm out!"), output -1.
- If a bidding price is mentioned (e.g., saying "I bid $xxx!"), output that price number (e.g., $xxx).
Here is the response:

{response}

Don't say anything else other than just a number: either the bidding price (e.g., $xxx, with $) or -1.
""".strip()


AUCTION_HISTORY = """
## Auction Log

### 1. Equipment E, starting at $5000.

#### 1st bid:
* Bidder 1: $5500
* Bidder 2: $5100
* Bidder 3: $5100
* Bidder 4: $5500
* Bidder 5: $6000

#### 2nd bid:
* Bidder 1: Withdrew
* Bidder 2: Withdrew
* Bidder 3: Withdrew
* Bidder 4: $6500

#### 3rd bid:
* Bidder 5: $7000

#### 4th bid:
* Bidder 4: Withdrew

#### Hammer price (true value):
* Bidder 5: $7000 ($10000)

### 2. Thingamajig C, starting at $1000.

#### 1st bid:
* Bidder 1: $1500
* Bidder 2: Withdrew
* Bidder 3: Withdrew
* Bidder 4: Withdrew
* Bidder 5: Withdrew

#### Hammer price (true value):
* Bidder 1: $1500 ($2000)

### 3. Component S, starting at $1000.

#### 1st bid:
* Bidder 1: $1200
* Bidder 2: $1050
* Bidder 3: $1000
* Bidder 4: Withdrew
* Bidder 5: $1200

#### 2nd bid:
* Bidder 2: Withdrew
* Bidder 3: $1300
* Bidder 5: $1300

#### 3rd bid:
* Bidder 1: Withdrew
* Bidder 3: $1400

#### 4th bid:
* Bidder 5: Withdrew

#### Hammer price (true value):
* Bidder 3: $1400 ($2000)

### 4. Implement G, starting at $1000.

#### 1st bid:
* Bidder 1: $1100
* Bidder 2: $1000
* Bidder 3: $1100
* Bidder 4: Withdrew
* Bidder 5: $1500

#### 2nd bid:
* Bidder 1: Withdrew
* Bidder 2: Withdrew
* Bidder 3: $1600

#### 3rd bid:
* Bidder 5: $1700

#### 4th bid:
* Bidder 3: Withdrew

#### Hammer price (true value):
* Bidder 5: $1700 ($2000)

### 5. Piece T, starting at $1000.

#### 1st bid:
* Bidder 1: $1100
* Bidder 2: $1000
* Bidder 3: $1100
* Bidder 4: Withdrew
* Bidder 5: $1200

#### 2nd bid:
* Bidder 1: Withdrew
* Bidder 2: $1300
* Bidder 3: $1300

#### 3rd bid:
* Bidder 2: $1400
* Bidder 5: Withdrew

#### 4th bid:
* Bidder 3: $1500

#### 5th bid:
* Bidder 2: Withdrew

#### Hammer price (true value):
* Bidder 3: $1500 ($2000)

### 6. Doodad D, starting at $1000.

#### 1st bid:
* Bidder 1: Withdrew
* Bidder 2: $1000
* Bidder 3: Withdrew
* Bidder 4: $1010
* Bidder 5: $1300

#### 2nd bid:
* Bidder 2: Withdrew
* Bidder 4: Withdrew

#### Hammer price (true value):
* Bidder 5: $1300 ($2000)

### 7. Gizmo F, starting at $1000.

#### 1st bid:
* Bidder 1: $1100
* Bidder 2: $1000
* Bidder 3: Withdrew
* Bidder 4: Withdrew
* Bidder 5: Withdrew

#### 2nd bid:
* Bidder 2: $1200

#### 3rd bid:
* Bidder 1: Withdrew

#### Hammer price (true value):
* Bidder 2: $1200 ($2000)

### 8. Widget A, starting at $1000.

#### 1st bid:
* Bidder 1: $2200
* Bidder 2: $1000
* Bidder 3: $1100
* Bidder 4: Withdrew
* Bidder 5: Withdrew

#### 2nd bid:
* Bidder 2: Withdrew
* Bidder 3: Withdrew

#### Hammer price (true value):
* Bidder 1: $2200 ($2000)

### 9. Gadget B, starting at $1000.

#### 1st bid:
* Bidder 1: $1200
* Bidder 2: Withdrew
* Bidder 3: Withdrew
* Bidder 4: $1000
* Bidder 5: Withdrew

#### 2nd bid:
* Bidder 4: Withdrew

#### Hammer price (true value):
* Bidder 1: $1200 ($2000)

### 10. Mechanism J, starting at $5000.

#### 1st bid:
* Bidder 1: Withdrew
* Bidder 2: $5000
* Bidder 3: $5100
* Bidder 4: $6000
* Bidder 5: Withdrew

#### 2nd bid:
* Bidder 2: $6500
* Bidder 3: $6500

#### 3rd bid:
* Bidder 3: $7000
* Bidder 4: $7000

#### 4th bid:
* Bidder 2: $7500
* Bidder 3: Withdrew

#### 5th bid:
* Bidder 4: $8000

#### 6th bid:
* Bidder 2: $8500

#### 7th bid:
* Bidder 4: Withdrew

#### Hammer price (true value):
* Bidder 2: $8500 ($10000)

## Personal Report

* Bidder 1, starting with $10000, has won 3 items in this auction, with a total profit of $1100.:
  * Won Thingamajig C at $1500 over $1000, with a true value of $2000.
  * Won Widget A at $2200 over $1000, with a true value of $2000.
  * Won Gadget B at $1200 over $1000, with a true value of $2000.

* Bidder 2, starting with $10000, has won 2 items in this auction, with a total profit of $2300.:
  * Won Gizmo F at $1200 over $1000, with a true value of $2000.
  * Won Mechanism J at $8500 over $5000, with a true value of $10000.

* Bidder 3, starting with $10000, has won 2 items in this auction, with a total profit of $1100.:
  * Won Component S at $1400 over $1000, with a true value of $2000.
  * Won Piece T at $1500 over $1000, with a true value of $2000.

* Bidder 4, starting with $10000, has won 0 items in this auction, with a total profit of $0.:

* Bidder 5, starting with $10000, has won 3 items in this auction, with a total profit of $4000.:
  * Won Equipment E at $7000 over $5000, with a true value of $10000.
  * Won Implement G at $1700 over $1000, with a true value of $2000.
  * Won Doodad D at $1300 over $1000, with a true value of $2000.
""".strip()
