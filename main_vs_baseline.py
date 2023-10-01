import json
import os
from pathlib import Path

import inquirer
import typer
from rich.console import Console
from rich.prompt import IntPrompt, Prompt, Confirm
import argparse
import logging

import util
from model import get_all_embeddings, get_all_llms
from setting import Settings, get_all_model_settings, load_model_setting
# import Settings, get_all_model_settings, load_model_setting
from model import agi_init
import gym
from retriever import (
    create_new_memory_retriever,
)
import random
from rlcard.utils import set_seed
import rlcard
from rlcard import models
from rlcard.models import leducholdem_rule_models

console = Console()


logging_dict = {}


    # return logger
def run(args):
    """
    Run IIG-AGI
    """
    settings = Settings()

    settings.model = load_model_setting(args.llm)

    # Model initialization verification
    res = util.verify_model_initialization(settings)
    if res != "OK":
        console.print(res, style="red")
        return
    # Get inputs from the user
    agent_count = args.agents_num
    # if agent_count < 2:
    #     Console.print("Please config at least 2 agents, exiting", style="red")
    #     return
    agent_configs = []
    agent_names = []
    for idx in range(agent_count):
       
        agent_config = {}
        while True:
            agent_file = args.player1_config if idx == 0 else args.player2_config
            if not os.path.isfile(agent_file):
                console.print(f"Invalid file path: {agent_file}", style="red")
                continue
            try:
                agent_config = util.load_json(Path(agent_file))
                agent_config["path"] = agent_file
                if agent_config == {}:
                    console.print(
                        "Empty configuration, please provide a valid one", style="red"
                    )
                    continue
                break
            except json.JSONDecodeError:
                console.print(
                    "Invalid configuration, please provide a valid one", style="red"
                )
                agent_file = Prompt.ask(
                    "Enter the path to the agent configuration file", default="./agent.json"
                )
                continue
        agent_configs.append(agent_config)
        agent_names.append(agent_config["name"])
    while True:
        if not os.path.isfile(args.game_config):
            console.print(f"Invalid file path: {args.game_config}", style="red")
            continue
        try:
            game_config = util.load_json(Path(args.game_config))
            game_config["path"] = args.game_config
            if game_config == {}:
                console.print(
                    "Empty configuration, please provide a valid one", style="red"
                )
                continue
            break
        except json.JSONDecodeError:
            console.print(
                "Invalid configuration, please provide a valid one", style="red"
            )
            game_config = Prompt.ask(
                "Enter the path to the agent configuration file", default="./agent.json"
            )
            continue

    ctx = agi_init(agent_configs, game_config, console, settings, args.user_index)
    log_file_name = ctx.robot_agents[(args.user_index+1) % args.agents_num].name+'_vs_'+ctx.robot_agents[(args.user_index ) % args.agents_num].name + '_'+args.rule_model + '_'+args.llm+'_'+args.mode



    env = rlcard.make('leduc-holdem',config={'seed': args.seed})
    env.reset()

    # actions = ["continue", "interview", "exit"]
    bot_experience_summ = []
    for i in range(args.agents_num):
        bot_experience_summ.append([])
    # experience = ''
    chips = [50,50]
    # rule_model = leducholdem_rule_models.LeducHoldemRuleAgentV1()
    print('./memory_data/'+log_file_name + '_long_memory_summary'+'.json')
    start_num = 0

    if args.rule_model == 'cfr':
        rule_model = models.load('leduc-holdem-cfr').agents[0]
    else:
        import torch
        rule_model = torch.load(os.path.join('./models', 'leduc_holdem_'+args.rule_model+'_result/model.pth'), map_location='cpu')
        rule_model.set_device('cpu')
    pattern = ''

    for game_idx in range(start_num,args.game_num):
        bot_long_memory = []
        bot_short_memory = []
        for i in range(args.agents_num):
            bot_short_memory.append([f'{game_idx+1}th Game Start'])
            bot_long_memory.append([f'{game_idx + 1}th Game Start'])
        if args.random_seed:
            set_seed(random.randint(0,10000))
        else:
            set_seed(args.seed)
        env.reset()
        round = 0
        plan = ''
        while not env.is_over():
            # if from_checkpoint:

                # print("Turn:" + str(env.game.whose_turn))
                console.print("Player:" + str(env.get_player_id()))
                idx = env.get_player_id()
                if round == 0:
                    start_idx = idx
                if args.user_index == idx and args.user:
                    console.print(env.get_state(env.get_player_id())['raw_obs'], style="green")
                    act,_ = rule_model.eval_step(env.get_state(env.get_player_id()))
                    act = env._decode_action(act)
                    util.get_logging(logger_name=log_file_name + '_opponent_obs',
                        content={str(game_idx + 1) + "_" + str(round): {"raw_obs": env.get_state(env.get_player_id())['raw_obs']}})
                    util.get_logging(logger_name= log_file_name + '_opponent_act',
                                        content={str(game_idx + 1) + "_" + str(round): {
                                            "act": str(act), "talk_sentence": str("")}})
                    console.print(act, style="green")
                    bot_short_memory[(args.user_index + 1) % args.agents_num].append(
                        f"The valid action list of {ctx.robot_agents[args.user_index].name} is {env.get_state(env.get_player_id())['raw_legal_actions']}, and he tries to take action: {act}.")
                    if args.no_hindsight_obs:
                        bot_long_memory[(args.user_index) % args.agents_num].append(
                              f"{ctx.robot_agents[args.user_index].name} try to take action: {act}.")
                    else:
                        bot_long_memory[(args.user_index) % args.agents_num].append(
                         f"{ctx.robot_agents[args.user_index].name} have the observation: {env.get_state(env.get_player_id())['raw_obs']}, and try to take action: {act}.")

                else:
                    amy = ctx.robot_agents[idx]
                    amy_index = env.get_player_id()
                    amy_obs = env.get_state(env.get_player_id())['raw_obs']
                    amy_obs['game_num'] = game_idx+1
                    amy_obs['rest_chips'] = chips[idx]
                    amy_obs['opponent_rest_chips'] = chips[(idx+1)%args.agents_num]
                    valid_action_list = env.get_state(env.get_player_id())['raw_legal_actions']
                    opponent_name  = ctx.robot_agents[(idx+1)%args.agents_num].name
                    print(opponent_name) 

                    if  args.verbose_print:
                        console.print(amy_obs, style="green")

                    act, comm, bot_short_memory, bot_long_memory = amy.make_act(amy_obs,opponent_name, amy_index,valid_action_list, verbose_print= args.verbose_print,
                                                                                game_idx = game_idx,round=round,bot_short_memory=bot_short_memory, bot_long_memory=bot_long_memory, console=console,
                                                                                log_file_name=log_file_name,mode=args.mode)
        

                env.step(act,raw_action=True)
                round += 1
        pay_offs = env.get_payoffs()
        for idx in range(len(pay_offs)):
            pay_offs[idx] = pay_offs[idx]*2
            chips[idx] += pay_offs[idx]
        print(pay_offs)
        if pay_offs[0] > 0:
            win_message = f'{ctx.robot_agents[0].name} win {pay_offs[0]} chips, {ctx.robot_agents[1].name} lose {pay_offs[0]} chips'
        else:
            win_message = f'{ctx.robot_agents[1].name} win {pay_offs[1]} chips, {ctx.robot_agents[0].name} lose {pay_offs[1]} chips'
        print(win_message)
        bot_short_memory[0].append(win_message)
        bot_short_memory[1].append(win_message)
        bot_long_memory[0].append(win_message)
        bot_long_memory[1].append(win_message)

        for i in range(args.agents_num):
            long_memory = '\n'.join(
                [x + '\n' + y for x, y in zip(bot_long_memory[start_idx], bot_long_memory[(start_idx + 1) % args.agents_num])])
            # print(long_memory)
            memory_summarization = ctx.robot_agents[(args.user_index + 1) % args.agents_num].get_summarization(
                ctx.robot_agents[(args.user_index + 1) % args.agents_num].name, long_memory, ctx.robot_agents[(args.user_index) % args.agents_num].name,no_highsight_obs=args.no_hindsight_obs)
            
            ctx.robot_agents[i].add_long_memory(f"{game_idx+1}th Game Start! \n"+memory_summarization)
            # if args.
            if i != args.user_index:
                util.get_logging(logger_name= log_file_name + '_long_memory',
                            content={str(game_idx + 1): {"long_memory": long_memory}})
                util.get_logging(logger_name= log_file_name + '_long_memory_summary',
                        content={str(game_idx + 1): {"long_memory_summary": memory_summarization}})




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Suspicion Agent',
        description='Playing Imperfect Information Games with LLM',
        epilog='Text at the bottom of help')
    parser.add_argument("--player1_config", default="./person_config/Persuader.json", help="experiments name")
    parser.add_argument("--player2_config", default="./person_config/GoodGuy.json", help="experiments name")
    parser.add_argument("--game_config", default="./game_config/leduc_limit.json", help="./game_config/leduc_limit.json, ./game_config/limit_holdem.json, ./game_config/coup.json")
    parser.add_argument("--seed", type=int, default=1, help="random_seed")
    parser.add_argument("--llm", default="openai-gpt-4-0613", help="environment flag, openai-gpt-4-0613 or openai-gpt-3.5-turbo")
    parser.add_argument("--rule_model", default="cfr", help="rule model: cfr or nfsp or dqn or dmc")
    parser.add_argument("--mode", default="second_tom", help="inference mode: normal or first_tom or second_tom")
    parser.add_argument("--agents_num", type=int, default=2)
    parser.add_argument("--user", action="store_true", help="one of the agents is baseline mode, e.g. cfr, nfsp")
    parser.add_argument("--verbose_print", action="store_true")
    parser.add_argument("--user_index", type=int, default=1, help="user position: 0 or 1")
    parser.add_argument("--game_num", type=int, default=50)
    parser.add_argument("--random_seed", action="store_true")
    parser.add_argument("--no_hindsight_obs", action="store_true")

    args = parser.parse_args()
    run(args)

