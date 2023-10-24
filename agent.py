# Reference: https://python.langchain.com/en/latest/use_cases/agent_simulations

import re
from datetime import datetime
from typing import List, Optional, Tuple

from langchain import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from pydantic.v1 import BaseModel, Field
from termcolor import colored
import util
import time

class SuspicionAgent(BaseModel):
    """A character with memory and innate characteristics."""

    name: str
    game_name: str
    age: int
    observation_rule: str
    """The traits of the character you wish not to change."""
    status: str
    """Current activities of the character."""
    llm: BaseLanguageModel

    """The retriever to fetch related memories."""
    verbose: bool = False

    reflection_threshold: Optional[float] = None
    """When the total 'importance' of memories exceeds the above threshold, stop to reflect."""

    current_plan: List[str] = []
    belief: str = ""
    pattern: str = ""
    long_belief: str = ""
    counter_belief: str = ""
    plan: str = ""
    high_plan: str = ""
    """The current plan of the agent."""

    memory: List = ['']
    summary: str = ""  #: :meta private:
    summary_refresh_seconds: int = 3600  #: :meta private:
    last_refreshed: datetime = Field(default_factory=datetime.now)  #: :meta private:

    memory_importance: float = 0.0  #: :meta private:
    max_tokens_limit: int = 1200  #: :meta private:
    read_observation: str = ""  #: :meta private:

    rule: str = ""  #: :meta private:
    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True


        

    
    
    def add_long_memory(self, memory_content: str) -> List[str]:
        """Add an observation or memory to the agent's memory."""
        self.memory.append(memory_content)
        return  self.memory


 

    def planning_module(self, observation: str,  recipient_name:str, previous_conversation: List[str] =None, belief: str =None, valid_action_list: List[str] = None, short_memory_summary:str = "",pattern:str = "",last_plan:str = "", mode: str = "second_tom") -> str:
        """Make Plans and Evaluate Plans."""
        """Combining these two modules together to save costs"""

        if mode == 'second_tom':
            prompt = PromptTemplate.from_template(
                "You are the objective player behind a NPC character called {initiator_name}, and you are playing the board game {game_name} with {recipient_name}.\n"
            + " The game rule is: {rule} \n"
             +'{pattern}\n'
              + " Your observation about the game status now is: {observation}\n"
            +'{belief}\n'
            + " Understanding all given information, can you do following things:"
            + " Make Reasonable Plans: Please plan several strategies according to actions {valid_action_list} you can play now to win the finally whole {game_name} games step by step. Note that you can say something or keep silent to confuse your opponent. " 
                 + " Potential {recipient_name}'s actions (if release) and Estimate Winning/Lose/Draw Rate for Each Plan: From the perspective of {recipient_name} , please infer what the action {recipient_name} with probability (normalize to number 100% in total) would do when {recipient_name} holds different cards and then calculate the winning/lose/draw rates when {recipient_name} holds different cards step by step. At last, please calculate the overall winning/lose/draw rates for each plan step by step considering  {recipient_name}'s behaviour pattern. Output in a tree-structure: "        
                + "Output: Plan 1:  If I execute plan1.  "
                          "The winning/lose/draw rates when {recipient_name} holds card1: Based on {recipient_name}'s behaviour pattern, In the xx round, because {recipient_name} holds card1  (probability) and the combination with current public card (if release)  (based on my belief on {recipient_name}), and if he sees my action, {recipient_name} will do action1 (probability) ( I actually hold card and the public card (if reveal) is , he holds card1 and the public card (if reveal), considering Single Game Win/Draw/Lose Rule, please infer I will win/draw/lose  step by step ), action2 (probability) (considering Single Game Win/Draw/Lose Rule, please infer I will win/draw/lose step by step  ),.. (normalize to number 100% in total); \n   Overall (winning rate for his card1) is (probability = his card probability * win action probability), (lose rate for his card2) is (probability= his card probability * lose action probability), (draw rate for his card2) is (probability = his card probability * draw action probability)  "  
                          "The winning/lose/draw rates when {recipient_name} holds card2: Based on {recipient_name}'s behaviour pattern, In the xx round, because {recipient_name} holds card2  (probability) and the combination with current public card (if release)  (based on my belief on {recipient_name}) ,  and if he sees my action, he will do action1 (probability) (I actually hold card and the public card (if reveal) is , he holds card1 and the public card (if reveal), considering Single Game Win/Draw/Lose Rule, please infer I will win/draw/lose  step by step ).. action2 (probability) (normalize to number 100% in total) (considering Single Game Win/Draw/Lose Rule, please infer I will win/draw/lose step by step ),.. ;..... continue ....\n Overall (winning rate for his card2) is (probability = his card probability * win action probability), (lose rate for his card2) is (probability= his card probability * lose action probability), (draw rate for his card2) is (probability = his card probability * draw action probability) "  
                          "...\n"
                          "Plan1 overall {initiator_name}'s Winning/Lose/Draw rates : the Winning rate (probability) for plan 1 is (winning rate for his card1) + (winning rate for his card2) + .. ; Lose rate (probability) for plan 1 : (lose rate for his card1) + (lose rate for his card2) + .. ; Draw Rate (probability) for plan 1  : (draw rate for his card1) + (draw rate for his card2) + ... ;  (normalize to number 100% in total) for plan1 \n"
                "Plan 2: If I execute plan2, The winning/lose/draw rates when {recipient_name} holds card1: Based on {recipient_name}'s behaviour pattern, In the xx round, if {recipient_name} holds card1  (probability)  and the combination with current public card (if release),  .. (format is similar with before ) ... continue .."
                "Plan 3: .. Coninue ... "
                + " The number of payoffs for each plan: Understanding your current observation,  each new plans, please infer the number of wininng/lose payoffs for each plan step by step, Output: Plan1: After the action, All chips  in the pot:  If win, the winning payoff would be (Calculated by Winning Payoff Rules step by step) :  After the action,  All chips in the pot:  If lose , the lose payoff would be:  (Calculated by Lose Payoff Rules step by step). Plan2:  After the action, All chips in the pot:  If win, the winning chips would be (Calculated by Winning Payoff Rules step by step):  After the action, All chips in the pot:  If lose , the lose chips would be:  (Calculated by Lose Payoff Rules step by step). If the number of my chips in pots have no change, please directly output them. \n"
                + " Estimate Expected Chips Gain for Each Plan: Understanding all the information and Estimate Winning/Lose/Draw Rate for Each Plan, please estimate the overall average Expected Chips Gain for each plan/strategy in the current game by calculating winning rate * (Winning Payoff Rule in the game rule) - lose rate * (Lose Payoff Rule in the game rule) step by step"
                + " Plan Selection: Please output the rank of estimated expected chips gains for every plan objectively step by step, and select the plan/strategy with the highest estimated expected chips gain considering both the strategy improvement. \n "
            )
       
        elif mode == 'first_tom':
            prompt = PromptTemplate.from_template(
                "You are the player behind a NPC character called {initiator_name}, and you are playing the board game {game_name} with {recipient_name}.\n"
            + " The game rule is: {rule} \n"
            + " {pattern} \n"
            + " Your observation about the game status now is: {observation}\n"
            + ' {belief}\n'
            + " Understanding all given information, can you do following things:"
            + " Make Reasonable Plans: Please plan several strategies according to actions {valid_action_list} you can play now to win the finally whole {game_name} games step by step. Note that you can say something or keep silent to confuse your opponent." 
            + " Potential {recipient_name}'s actions and Estimate Winning/Lose/Draw Rate: From the perspective of {recipient_name}, please infer what the action {recipient_name} with probability (normalize to number 100% in total) would do when {recipient_name} holds different cards, and then calculate the winning/lose/draw rates when {recipient_name} holds different cards step by step. Output in a tree-structure: "        
                + "Output: Based on {recipient_name}'s behaviour pattern and Analysis on {recipient_name}'s cards, "
                "Winning/lose/draw rates when {recipient_name} holds card1 in the xx round,: if {recipient_name} holds card1  (probability) (based on my belief on {recipient_name}) with the public card  (if release), {recipient_name} will do action1 (probability) (infer I will win/draw/lose step by step (considering Single Game Win/Draw/Lose Rule and my factual card analysis with public card (if release), his card analysis with public card (if release) step by step ), action2 (probability) (infer I will win/draw/lose step by step  ),.. (normalize to number 100% in total);    Overall (winning rate for his card1) is (probability = his card probability * win action probability), (lose rate for his card2) is (probability= his card probability * lose action probability), (draw rate for his card2) is (probability = his card probability * draw action probability)  "  
                          "The winning/lose/draw rates when {recipient_name} holds card2 in the xx round,: If {recipient_name} holds card2  (probability) (based on my belief on {recipient_name}) with the public card  (if release),  he will do action1 (probability) (infer I will win/draw/lose (considering Single Game Win/Draw/Lose Rule and my factual card analysis with current public card (if release), his card analysis with current public card (if release)) step by step ).. action2 (probability) (normalize to number 100% in total) (infer I will win/draw/lose step by step ),..  based on  {recipient_name}'s behaviour pattern;..... continue .... Overall (winning rate for his card2) is (probability = his card probability * win action probability), (lose rate for his card2) is (probability= his card probability * lose action probability), (draw rate for his card2) is (probability = his card probability * draw action probability) "  
                          "..."
                          "Overall {initiator_name}'s Winning/Lose/Draw rates : Based on the above analysis,  the Winning rate (probability) is (winning rate for his card1) + (winning rate for his card2) + .. ; Lose rate (probability): (lose rate for his card1) + (lose rate for his card2) + .. ; Draw Rate (probability): (draw rate for his card1) + (draw rate for his card2) + ... ;  (normalize to number 100% in total). \n"         
            + " Potential believes about the number of winning and lose payoffs for each plan: Understanding the game rule, your current observation, previous actions summarization, each new plans, Winning Payoff Rule,  Lose Payoff Rule, please infer your several believes about  the number of chips in pots for each plan step by step, Output: Plan1: Chips in the pot:  If win, the winning payoff would be (Calculated by Winning Payoff Rules in the game rule) :  After the action, If lose , the lose payoff would be: . Plan2:  Chips in the pot:  If win, the winning chips would be (Calculated by Winning Payoff Rules in the game rule):  After the action, If lose , the lose chips would be: . If the number of my chips in pots have no change, please directly output them. "
            + " Estimate Expected Chips Gain for Each Plan: Understanding the game rule, plans,  and your knowledge about the {game_name}, please estimate the overall average Expected Chips Gain for each plan/strategy in the current game by calculating winning rate * (Winning Payoff Rule in the game rule) - lose rate * (Lose Payoff Rule in the game rule)., explain what is the results if you do not select the plan, and explain why is this final  Expected  Chips Gain reasonablely step by step? "
            + " Plan Selection: Please output the rank of estimated expected chips gains for every plan objectively step by step, and select the plan/strategy with the highest estimated expected chips gain considering both the strategy improvement. \n\n "
                )
        else:
             prompt = PromptTemplate.from_template(
                "You are the player behind a NPC character called {initiator_name}, and you are playing the board game {game_name} with {recipient_name}.\n"
            + " The game rule is: {rule} \n"
            + "  {pattern} \n"
            + " Your observation about the game status now is: {observation}\n"
            + " Understanding all given information, can you do following things:"
            + " Make Reasonable Plans: Please plan several strategies according to actions {valid_action_list} you can play now to win the finally whole {game_name} games step by step. Note that you can say something or keep silent to confuse your opponent." 
               + " Estimate Winning/Lose/Draw Rate for Each Plan: Understanding the given information, and your knowledge about the {game_name}, please estimate the success rate of each step of each plan step by step and the overall average winning/lose/draw rate  (normalize to number 100% in total) of each plan/strategy for the current game  step by step following the templete: If I do plan1, because I hold card, the public information (if release) and Single Game Win/Draw/Lose Rule, I will win or Lose or draw (probability);  ... continue  .... Overall win/draw/lose rate: Based on the analysis, I can do the weighted average step by step to get that the overall weighted average winning rate is (probability), average lose rate is (probability), draw rate is (probability) (normalize to number 100% in total)\n "
               + " Potential believes about the number of winning and lose payoffs for each plan: Understanding the game rule, your current observation, previous actions summarization, each new plans, Winning Payoff Rule,  Lose Payoff Rule, please infer your several believes about  the number of chips in pots for each plan step by step, Output: Plan1: Chips in the pot:  If win, the winning payoff would be (Calculated by Winning Payoff Rules in the game rule) :  After the action,  Chips in the pot:  If lose , the lose payoff would be: . Plan2:  Chips in the pot:  If win, the winning chips would be (Calculated by Winning Payoff Rules in the game rule):  After the action, Chips in the pot:   If lose , the lose chips would be: . If the number of my chips in pots have no change, please directly output them. "
               +" Estimate Expected Chips Gain for Each Plan: Understanding the game rule, plans,  and your knowledge about the {game_name}, please estimate the overall average Expected Chips Gain for each plan/strategy in the current game by calculating winning rate * (Winning Payoff Rule in the game rule) - lose rate * (Lose Payoff Rule in the game rule)., explain what is the results if you do not select the plan, and explain why is this final  Expected  Chips Gain reasonablely step by step? "
            + " Plan Selection: Please output the rank of estimated expected chips gains for every plan objectively step by step, and select the plan/strategy with the highest estimated expected chips gain considering both the strategy improvement. \n\n "
                )

        agent_summary_description = short_memory_summary
       
        belief = self.belief if belief is None else belief
      
        kwargs = dict(
            
            recent_observations=agent_summary_description,
            last_plan=last_plan,
            belief=belief,
            initiator_name=self.name,
            pattern=pattern,
            recipient_name=recipient_name,
            observation=observation,
            rule=self.rule,
            game_name=self.game_name,
            valid_action_list=valid_action_list
        )


        plan_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)
        self.plan = plan_prediction_chain.run(**kwargs)
        self.plan = self.plan.strip()

        return self.plan.strip()

       
    
    def get_belief(self, observation: str, recipient_name: str,short_memory_summary:str,pattern:str = "",mode: str = "second_tom") -> str:
        """React to get a belief."""
        if mode == 'second_tom':
            prompt = PromptTemplate.from_template(
                "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} with {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                + " Your estimated judgement about the behaviour pattern of {recipient_name} and improved strategy is: {pattern} \n"
                + " Your observation now is: {observation}\n"
                + " Your current game progress summarization including actions and conversations with {recipient_name} is: {recent_observations}\n"
                + " Understanding the game rule, the cards you have, your observation,  progress summarization in the current game, the estimated behaviour pattern of {recipient_name}, the potential guess pattern of {recipient_name} on you, and your knowledge about the {game_name}, can you do following things? "
                + " Analysis on my Cards: Understanding all given information and your knowledge about the {game_name}, please analysis what is your best combination and advantages of your cards in the current round step by step." 
                + " Belief on {recipient_name}'s cards: Understanding all given information, please infer the probabilities about the cards of {recipient_name}  (normalize to number 100% in total) objectively step by step." 
                "Output: {recipient_name}  saw my history actions (or not) and then did action1 (probability) in the 1st round , ... continue..... Before this round, {recipient_name}  say my history actions (or not) and  did action1 (probability), because {recipient_name}'s behaviour pattern and the match with the public card (if release), {recipient_name} tends to have card1 (probability), card2 (probability) ..continue.. (normalize to number 100% in total)."
                + " Analysis on {recipient_name}'s Cards: Understanding all given information and your knowledge about the {game_name}, please analysis what is {recipient_name}'s best combination and advantages of {recipient_name}'s cards in the current round  step by step." 
                + " Potential {recipient_name}'s current believes about your cards: Understanding all given information and your knowledge about the {game_name}, If you were {recipient_name} (he can only observe my actions but cannot see my cards), please infer the {recipient_name}'s  believes about your cards with probability (normalize to number 100% in total) step by step. Output: {agent_name} did action1 (probability) (after I did action or not) in the 1st round, , ... continue...  {agent_name} did action1 (probability) (after I did action or not)  in the current round,, from the perspective of {recipient_name}, {agent_name} tends to have card1 (probability), card2 (probability) ... (normalize to number 100% in total) ."
                )
        elif mode == 'first_tom':
            prompt = PromptTemplate.from_template(
                "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} with {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                + " Your estimated judgement about the behaviour pattern of {recipient_name} and improved strategy is: {pattern} \n"
                + " Your observation now is: {observation}\n"
                + " Your current game progress summarization including actions and conversations with {recipient_name} is: {recent_observations}\n"
                + " Understanding the game rule, the cards you have, your observation,  progress summarization in the current game, the estimated behaviour pattern of {recipient_name}, and your knowledge about the {game_name}, can you do following things? "
                + " Analysis on my Cards: Understanding all given information, please analysis what is your best combination and advantages of your cards  in the current round  step by step." 
                + " Belief on {recipient_name}'s cards: Understanding all given information, please infer your the probabilities about the cards of {recipient_name}  (normalize to number 100% total)  step by step. Templete: In the 1st round, {recipient_name} did action1 (probability),  ... continue... In the current round, {recipient_name} did action1 (probability), because {recipient_name}'s behaviour pattern and the match with the current public card (if release), he tends to have card1 (probability), card2 (probability) (normalize to number 100% in total). "
            + " Analysis on {recipient_name}'s Cards: Understanding all given information, please analysis what is {recipient_name}'s best combination and advantages of {recipient_name}'s cards  in the current round  step by step." 
                
            )
        agent_summary_description = short_memory_summary

        kwargs = dict(
            agent_summary_description=agent_summary_description,
            recent_observations=agent_summary_description,
            agent_name=self.name,
            pattern= pattern,
            recipient_name=recipient_name,
            observation=observation,
            game_name=self.game_name,
            rule=self.rule

        )
        print(recipient_name)
        
        belief_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)
        self.belief = belief_prediction_chain.run(**kwargs)
        self.belief = self.belief.strip()
        return self.belief.strip()

    
    def get_pattern(self, recipient_name: str,game_pattern: str='', last_k:int=20,short_summarization:str='',mode:str='second_tom') -> str:
        """React to get a belief."""
       
        if mode == 'second_tom':
            prompt = PromptTemplate.from_template(
                "You are the objective player behind a NPC character called {agent_name}, and you are playing {game_name} with {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                + " Your previous game memory including observations, actions and conversations with {recipient_name} is: {long_memory}\n"
                + "  {recipient_name}'s game pattern: Understanding all given information and your understanding about the {game_name}, please infer and estimate as many as possible reasonable {recipient_name}'s game behaviour pattern/preferences for each card he holds and each round with probability (normalize to number 100\% in total for each pattern item) and please also infer advantages of his card, and analysis how the {recipient_name}'s behaviour pattern/preferences are influenced by my actions when he holds different cards step by step. Output  as a tree-structure    " 
                + "Output: When {recipient_name} holds card1 and the combination of public card (if release):  if {recipient_name}  is the first to act, he would like to do action1 (probabilities), action2 (probabilities) ... continue ..    If {recipient_name} sees the action1/action2/action3 of the opponent or not, he would like to do action1 (probabilities), action2 (probabilities) ... continue ...  (normalize to number 100% in total), if {recipient_name} sees the action2 of the opponent  or not,  ... continue ..(more patterns with different actions)..  in the 1st round, ;  If {recipient_name} sees the action1 of the opponent  or not,  he would like to do action1 (probabilities), action2 (probabilities) ... continue... (normalize to number 100% in total), ...  continue ..(more patterns)..In the 2nd round,;"
                 "When {recipient_name} holds card2 and combination of public card (if release): if {recipient_name}  is the first to act, he would like to do action1 (probabilities), action2 (probabilities) ... continue .. If {recipient_name} sees the action1 of the opponent  or not, he would like to do action1 (probabilities), action2 (probabilities) .. continue ... (normalize to number 100% in total)...in the 1st round,; .. continue ..(more patterns  with different actions).in the 2nd round .. "
                 " (more patterns with different cards).. continue.."
                + "  {recipient_name}'s guess on my game pattern: Understanding all given information, please infer several reasonable believes about my game pattern/preference when holding different cards from the perspective of {recipient_name} (please  consider the advantages of the card, actions and the the match with the public card (if release)) for every round of the game in detail as a tree-structure output step by step"
               + "Output: In the 1st round, When name holds card1 with public card (if release), he would like to do (probabilities), action2 (probabilities)  (normalize to number 100% in total) o ... continue .. and then do action ...;"
                 "When name holds card2 with public card (if release), ... "
                 " .. continue.."
                + " Strategy Improvement: Understanding the above information, think about what strategies I can adopt to exploit the game pattern of {recipient_name} and {recipient_name}'s guess on my game pattern for winning {recipient_name} in the whole game step by step.  (Note that you cannot observe the cards of the opponent during the game, but you can observe his actions). Output as a tree-structure:"
                "When I hold card and the public card (if release), and see the action of the opponent, I would like to do action1; ... "
            )
        elif mode == 'first_tom':
            prompt = PromptTemplate.from_template(
                "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} with {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                + " Your previous game memory including observations, actions and conversations with {recipient_name} is: {long_memory}\n"
                + " Please understand the game rule, previous all game history and your knowledge about the {game_name}, can you do following things for future games? "
                + "  {recipient_name}'s game pattern: Understanding all given information, please infer all possible reasonable {recipient_name}'s game pattern/preferences for each card he holds and each round with probability (normalize to number 100\% in total for each pattern item) for every round of the game as a tree-structure output step by step  " 
                + "Output: In the 1st round, when name holds card1 and the public card (if release), he would like to do action (probabilities); when name holds card2 and the public card (if release), he would like to do action (probabilities), ... continue.. In the 2nd round,  when name holds card1 and the public card (if release), .(similar with before).. continue. "
               + " Number of chips reason: Think about why you can have these chips in all previous games step by step. "
                + " Reflex: Reflex which your actions are right or wrong in previous games to win or Lose conrete chips step by step  (Note that you cannot observe the cards of the opponent during the game, but you can observe his actions) "
                + " Strategy Improvement: Understanding the above information, think about what strategies I can adopt to exploit the game pattern of {recipient_name} for winning {recipient_name} in the whole game step by step.  (Note that you cannot observe the cards of the opponent during the game, but you can observe his actions). Output as a tree-structure:"
                )
        else:
            prompt = PromptTemplate.from_template(
                "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} with {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                + " Your previous game memory including observations, actions and conversations with {recipient_name} is: {long_memory}\n"
                + " Please understand the game rule, previous all game history and your knowledge about the {game_name}, can you do following things for future games? "
                + " Number of chips reason: Think about why you can have these chips in all previous games step by step. "
                + " Reflex: Reflex which your actions are right or wrong in previous games to win or Lose conrete chips step by step. (Note that you cannot observe the cards of the opponent during the game, but you can observe his actions) "
                + " Strategy Improvement: Understanding the above information, think about what strategies I need to adopt to win {recipient_name} for the whole game step by step.  (Note that you cannot observe the cards of the opponent during the game, but you can observe his actions). Output as a tree-structure:"
            )
        reflection_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        long_memory = self.memory[-last_k:]
        long_memory_str = "\n\n".join([o for o in long_memory])
        
        kwargs = dict(
            long_memory=long_memory_str,
            game_pattern=game_pattern,
            agent_name=self.name,
            recipient_name=recipient_name,
            game_name=self.game_name,
            rule=self.rule

        )
        # print(kwargs)

        self.long_belief = reflection_chain.run(**kwargs)
        self.long_belief = self.long_belief.strip()
        return self.long_belief.strip()



    def get_summarization(self, recipient_name: str,game_memory: str, opponent_name:str,no_highsight_obs:bool) -> str:
        """Get a long memory summarization to save costs."""
        if no_highsight_obs:
            prompt = PromptTemplate.from_template(
                "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} with {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                + " The observation conversion rules are: {observation_rule}\n"
                + " One game memory including observations, actions and conversations with {recipient_name} is: {long_memory}\n"
                + " Understanding the game rule, observation conversion rules and game history and your knowledge about the {game_name}, can you do following things:"
                + " History summarization: summary the game history with action, observation, and results information? using the templete, and respond shortly: In the first round of first game, name holds card1 does action .... continue ..." 
                + "{opponent_name}'s card reasoning: If the card of {opponent_name} is not available, because {agent_name}'s  card is xx and public card (if release) is xxx, and {opponent_name} behaviours are xx, the current game result is xx,  please  infer {opponent_name}'s card with probability (100% in total) with your understanding about the above all information confidently step by step. \n"
                )
        else:
            prompt = PromptTemplate.from_template(
                "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} with {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                + " The observation conversion rules are: {observation_rule}\n"
                + " One game memory including observations, actions and conversations with {recipient_name} is: {long_memory}\n"
                + " Understanding the game rule, observation conversion rules and game history and your knowledge about the {game_name}, can you do following things:"
                + " History summarization: summary the game history with action, observation, and results information? using the templete, and respond shortly: In the first round of first game, name holds card1 does action .... continue ..." 
                )       
        reflection_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        kwargs = dict(
            observation_rule=self.observation_rule,
            long_memory=game_memory,
            agent_name=self.name,
            recipient_name=recipient_name,
            opponent_name=opponent_name,
            # observation=observation,
            game_name=self.game_name,
            rule=self.rule

        )
        # print(kwargs)

        self.long_belief = reflection_chain.run(**kwargs)
        self.long_belief = self.long_belief.strip()
        return self.long_belief.strip()


    def get_short_memory_summary(self, observation: str, recipient_name: str,short_memory_summary:str) -> str:
        """React to get a belief."""
        prompt = PromptTemplate.from_template(
            "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} with {recipient_name}. \n"
            + " The game rule is: {rule} \n"
            + " Your current observation is: {observation}\n"
            + " The current game history including previous action, observations and conversation is: {agent_summary_description}\n"
            + " Based on the game rule, your observation and your knowledge about the {game_name}, please summarize the current history. Output as a tree-structure, and respond shortly: "
            + " In the first round, name does action, and say xxx .... continue ..." 
        )

        agent_summary_description = short_memory_summary
    
        kwargs = dict(
            agent_summary_description=agent_summary_description,
            recent_observations=agent_summary_description,
            agent_name=self.name,
            recipient_name=recipient_name,
            observation=observation,
            game_name=self.game_name,
            rule=self.rule

        )

        belief_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)
        self.belief = belief_prediction_chain.run(**kwargs)
        self.belief = self.belief.strip()
        return self.belief.strip()



    def convert_obs(self, observation: str, recipient_name: str, user_index: str, valid_action_list:str) ->  str:
        """React to get a belief."""
        prompt = PromptTemplate.from_template(
            "You are the player behind a NPC character called {agent_name} with player index {user_index}, and you are playing the board game {game_name} with {recipient_name}. \n"
            + " The game rule is: {rule} \n"
            + " Your observation now is: {observation}\n"
            + " You will receive a valid action list you can perform in this turn \n"
            + " Your valid action list is: {valid_action_list}\n"
            + " The observation conversion rules are: {observation_rule}\n"
            + " Please convert {observation} and {valid_action_list} to the readable text based on the observation conversion rules and your knowledge about the {game_name} (respond shortly).\n\n"
        )
        kwargs = dict(
            user_index=user_index,
            agent_name=self.name,
            rule=self.rule,
            recipient_name=recipient_name,
            observation=observation,
            valid_action_list=valid_action_list,
            game_name=self.game_name,
            observation_rule=self.observation_rule
        )
        obs_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)
        self.read_observation = obs_prediction_chain.run(**kwargs)
        self.read_observation = self.read_observation.strip()
        return self.read_observation



    def action_decision(self, observation: str, valid_action_list: List[str], promp_head: str, act: str = None,short_memory_summary:str="") -> Tuple[str,str]:
        """React to a given observation."""
        """React to a given observation."""
        prompt = PromptTemplate.from_template(
            promp_head
            + "\nYour plan is: {plan}"
            + "\n Based on the plan, please select the next action from the available action list: {valid_action_list} (Just one word) and say something to the opponent player to bluff or confuse him or keep silent to finally win the whole game and reduce the risk of your action (respond sentence only). Please respond them and split them by |"
            + "\n\n"
        )

        agent_summary_description = short_memory_summary

        kwargs = dict(
            agent_summary_description= agent_summary_description,
            # current_time=current_time_str,
            # relevant_memories=relevant_memories_str,
            agent_name= self.name,
            game_name=self.game_name,
            observation= observation,
            agent_status= self.status,
            valid_action_list = valid_action_list,
            plan = self.plan,
            belief = self.belief,
            act = act
        )
        action_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)

        result = action_prediction_chain.run(**kwargs)
        if "|" in result:
            result,result_comm = result.split("|",1)
        else:
            result_comm = ""
        return result.strip(),result_comm.strip()

    def make_act(self, observation: str,opponent_name: str, player_index:int,valid_action_list: List, verbose_print:bool,game_idx:int,round:int,bot_short_memory:List, bot_long_memory:List, console,log_file_name='', mode='second_tom',no_highsight_obs=False) -> Tuple[bool, str]:
        readable_text_amy_obs = self.convert_obs(observation, opponent_name, player_index, valid_action_list)
        if  verbose_print:
            console.print('readable_text_obs: ', style="red")
            print(readable_text_amy_obs)
                   
        time.sleep(0)
        if len(bot_short_memory[player_index]) == 1:
            short_memory_summary = f'{game_idx+1}th Game Start \n'+readable_text_amy_obs
        else:
            short_memory_summary = self.get_short_memory_summary(observation=readable_text_amy_obs, recipient_name=opponent_name,short_memory_summary='\n'.join(bot_short_memory[player_index]))

            
        if verbose_print:
            console.print('short_memory_summary: ', style="yellow")
            print(short_memory_summary)

        time.sleep(0)
        if  round <= 1:
                self.pattern = self.get_pattern(opponent_name,'',short_summarization=short_memory_summary,mode=mode)        
                console.print('pattern: ', style="blue")
                print(self.pattern)

        time.sleep(0)
        print(opponent_name)

        if mode == 'second_tom' or mode == 'first_tom':
            belief = self.get_belief(readable_text_amy_obs,opponent_name,short_memory_summary=short_memory_summary,pattern=self.pattern,mode=mode)
            if verbose_print:
                console.print(self.name + " belief: " , style="deep_pink3")
                print(self.name + " belief: " + str(belief))
                
        else:
            belief = ''

        time.sleep(0)
        plan = self.planning_module(readable_text_amy_obs,opponent_name, belief=belief,valid_action_list=valid_action_list,short_memory_summary=short_memory_summary,pattern=self.pattern,last_plan='', mode=mode)
        if  verbose_print:
            console.print(self.name + " plan: " , style="orchid")
            print(self.name + " plan: " + str(plan))
            
        time.sleep(0)
        promp_head = ''
        act, comm = self.action_decision(readable_text_amy_obs, valid_action_list, promp_head,short_memory_summary=short_memory_summary)

        if log_file_name is not None:
            util.get_logging(logger_name=log_file_name + '_obs',
                        content={str(game_idx + 1) + "_" + str(round): {"raw_obs": observation,
                                                                        "readable_text_obs": readable_text_amy_obs}})
            util.get_logging(logger_name=log_file_name + '_short_memory',
                        content={str(game_idx + 1) + "_" + str(round): {
                            "raw_short_memory": '\n'.join(bot_short_memory[player_index]),
                            "short_memory_summary": short_memory_summary}})
            util.get_logging(logger_name=log_file_name + '_pattern_model',
                                content={str(game_idx + 1) + "_" + str(round): self.pattern})
            util.get_logging(logger_name=log_file_name + '_belief',
                            content={str(game_idx + 1) + "_" + str(round): {
                                "belief": str(belief)}})
            util.get_logging(logger_name=log_file_name + '_plan',
                        content={str(game_idx + 1) + "_" + str(round): {
                            "plan": str(plan)}})
            util.get_logging(logger_name= log_file_name + '_act',
                        content={str(game_idx + 1) + "_" + str(round): {
                            "act": str(act), "talk_sentence": str(comm)}})
 

        while act not in valid_action_list:
            print('Action + ', str(act), ' is not a valid action in valid_action_list, please try again.\n')
            promp_head += 'Action {act} is not a valid action in {valid_action_list}, please try again.\n'
            act, comm = self.action_decision( readable_text_amy_obs, valid_action_list, promp_head,act)
        print(self.name + " act: " + str(act))
        print(comm)

        bot_short_memory[player_index].append(f"{self.name} have the observation {readable_text_amy_obs}, try to take action: {act} and say {comm} to {opponent_name}")
        bot_short_memory[((player_index + 1)%2)].append(f"{self.name} try to take action: {act} and say {comm} to {opponent_name}")

        bot_long_memory[player_index].append(
            f"{self.name} have the observation {observation}, try to take action: {act} and say {comm} to {opponent_name}")
        return act,comm,bot_short_memory,bot_long_memory
