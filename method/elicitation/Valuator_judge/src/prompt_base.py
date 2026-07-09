# for bidder
SYSTEM_MESSAGE = """
You are {name}, who is attending an art evaluation session as a judge. This session will have some other judges evaluating the same artworks. Each judge scores artworks independently on a scale of 0-100. The highest scorer for each item is recognized as the lead evaluator. Remember: {desire_desc}.

Here are some must-know rules for this evaluation:

1. Item Evaluation: You need to evaluate items based on the information provided (images, descriptions, context). Your evaluation reflects your subjective preference and aesthetic judgment. Different judges may have different scores for the same item.
2. Scoring: Score each artwork on a scale of 0-100. Higher scores mean higher aesthetic quality. The judge who gives the highest score is recognized as the lead evaluator for that item.
3. Consistency Constraint: Your scores should be consistent with your overall preference ranking. If you prefer artwork A over artwork B, then your score for A should be higher than your score for B.
4. Scoring Rounds: Each item has exactly 1 round of scoring. You must give your best score in this single round.
5. Mandatory Scoring: You MUST score every item. Abstaining is NOT allowed.
""".strip()


_LEARNING_STATEMENT = " and your learnings from previous auctions"


INSTRUCT_PLAN_TEMPLATE = """
As {name}, you are evaluating {item_num} artworks. This evaluation session has a total of {item_num} items to be sequentially presented, they are:
{items_info}

---

Please plan your evaluation strategy based on the information{learning_statement}. A well-thought-out plan helps you score consistently and fairly. Remember: {desire_desc}.

**Important Evaluation Guidelines:**
- Base your judgment ONLY on the aesthetic qualities, visual appeal, and context-fit of each item
- DO NOT infer value from starting prices - they are arbitrary and do not reflect true worth
- Different items may appeal to you differently based on your personal aesthetic preferences
- Consider how well each item fits its stated evaluation context
- Your scores should reflect a clear ranking: better artworks get higher scores

After articulating your thinking, assign a priority level to each item. Present the priorities for all items in a JSON format, each item should be represented as a key-value pair, where the key is the item name and the value is its priority on the scale from 1-3. An example output is: {{"Fixture Y": 3, "Module B": 2, "Product G": 2}}. The descriptions of the priority scale of items are as follows.
    * 1 - This item is the least appealing. You would score it low.
    * 2 - This item holds moderate appeal. You would score it in the mid range.
    * 3 - This item is of highest quality and most appealing. You would score it high.
""".strip()


INSTRUCT_BID_TEMPLATE = """
Now, the evaluator says: "{auctioneer_msg}"

**This is the only round for this item.** The highest scorer is recognized as the lead evaluator.

---

As {bidder_name}, you have to score this item on a scale of 0-100 based on your aesthetic evaluation{learning_statement}. Remember, {desire_desc}.

Here are some guidelines for scoring:
1. Evaluate your aesthetic preference for this item based on the image and description provided.
2. Consider the current highest score and decide what score YOU would give based on YOUR aesthetic evaluation.
3. **IMPORTANT: Your score must be consistent with your preference ranking.** If you ranked this item higher than another, your score should be higher too.
4. Think step by step about the artistic quality, composition, originality, and emotional impact.
5. Use the full 0-100 range to differentiate between artworks:
   - 80-100: Exceptional quality, top-tier artwork
   - 60-79: Good quality, above average
   - 40-59: Average quality
   - 20-39: Below average
   - 0-19: Poor quality

Give your reasons first, then make your final decision clearly. You MUST give a score (saying "I score xxx!" with a specific number from 0-100). Abstaining is not allowed.
""".strip()


INSTRUCT_SUMMARIZE_TEMPLATE = """
Here is the history of the scoring for {cur_item}:
"{bidding_history}"

The evaluator concludes: "{hammer_msg}"

---

{win_lose_msg}
As {bidder_name}, you have to update the status of the evaluation based on this round of scoring. Here's your previous status:
```
{prev_status}
```

Summarize the notable scoring patterns of all judges in this round. Then, update the status JSON regarding the following information:
- 'scores_given': The scores given for every item by each judge, listed as key-value pairs, for example, {{"judge_name": {{"item_name_1": score}}, {{"item_name_2": score}}, ...}}. If a judge hasn't scored any item yet, then the value for this judge should be an empty dictionary {{}}.
- Only include the judges mentioned in the given text. If a judge is not mentioned (e.g. Judge 4 in the following example), then do not include it in the JSON object.

After summarizing the scoring history, you must output the current status in a parsible JSON format. An example output looks like:
```
{{"scores_given": {{"Judge 1": {{"Item 2": 85, "Item 3": 72}}, "Judge 2": {{"Item 1": 90}}, "Judge 3": {{}}}}}}
```
""".strip()


INSTRUCT_LEARNING_TEMPLATE = """
Review and reflect on the historical data provided from a past evaluation session.

{past_auction_log}

Here are your past learnings:

{past_learnings}

Based on the evaluation log, formulate or update your learning points that could be advantageous to your strategies in the future. Focus on:
1. How well your scores aligned with your aesthetic preferences
2. Whether you effectively differentiated between artworks using the full 0-100 range
3. Patterns in other judges' scoring behavior
4. Strategic lessons about consistent and fair scoring

Your learnings should be strategic, of universal relevance, and practical use for future evaluations. Consolidate your learnings into a concise numbered list of sentences.
""".strip()


INSTRUCT_REPLAN_TEMPLATE = """
The current status of you and other judges is as follows:
```
{status_quo}
```

Here are the remaining items in the rest of the evaluation:
"{remaining_items_info}"

As {bidder_name}, considering the current status{learning_statement}, review your strategies. Adjust your plans based on the outcomes and new information to achieve your primary objective. This iterative process ensures that your approach remains relevant and effective. Please do the following:
1. Always remember: {desire_desc}.
2. Determine and explain if there's a need to update the priority list of remaining items based on the current status and your aesthetic preferences.
3. Present the updated priorities in a JSON format, each item should be represented as a key-value pair, where the key is the item name and the value is its priority on the scale from 1-3. An example output is: {{"Fixture Y": 3, "Module B": 2, "Product G": 2}}. The descriptions of the priority scale of items are as follows.
    * 1 - This item has lower aesthetic appeal to you. You would score it low.
    * 2 - This item has moderate aesthetic appeal but isn't a top priority. You would score it in the mid range.
    * 3 - This item has high aesthetic appeal and is a top priority. You would score it high.
""".strip()


# for auctioneer
PARSE_BID_INSTRUCTION = """
Your task is to parse a response from a judge in an art evaluation, and extract the score from the response. Here are the rules:
- If the judge abstains (e.g., saying "I abstain!"), output -1.
- If a score is mentioned (e.g., saying "I score 85!" or "My score is 85"), output that number.
Here is the response:

{response}

Don't say anything else other than just a number: either the score (0-100) or -1.
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
