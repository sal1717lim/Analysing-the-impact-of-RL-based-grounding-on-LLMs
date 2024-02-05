def generate_prompt(past_transitions, obs, info):
    prompt = "Possible actions of the agent: {}\n".format(", ".join(info["possible_actions"]))
    prompt += "{}\n".format(info["goal"])
    #prompt += "If you want to move forward, just say 'go' If you feel like turning right, you just need to say 'right' and if you want to turn left, you just need to say 'left' If you want to pick up an object in front of you, simply say 'pick' If you want to drop an object, say 'drop' And if you want to use an object to open a door, you just need to say 'toggle'.\n" 
    for transition in past_transitions:
        prompt += "Past Observation: {}\n".format(', '.join(transition["obs"]))
        prompt += "Past Action:{}\n".format(transition["act"])
    prompt += "Current Observation: {}\n".format(', '.join(obs))
    prompt += "Action:"
    return prompt
def prompt_template1(prompt):
    new_prompt=""
    _prompt=prompt.split("Observation")
    if len(_prompt)==2:
        new_prompt=_prompt[0]+"Observation"+_prompt[1]
    else:
        new_prompt+=_prompt[0]+"Observation"+_prompt[2]+"Observation"+_prompt[1]
        #new_prompt+="now:"+_prompt[1]+"your past:"+_prompt[0]
    
    return new_prompt
def prompt_template2(prompt):
    new_prompt=""
    _prompt=prompt.split("\n")
    new_prompt+=_prompt[1]+"\n"+_prompt[0]+"\n"
    for i in range(2,len(_prompt),1):
        new_prompt+=_prompt[i]+"\n"
    return new_prompt
def prompt_template3(prompt):
    new_prompt=""
    _prompt=prompt.split("\n")
    new_prompt+=_prompt[-2]+"\n"+_prompt[1]+"\n"
    new_prompt+=_prompt[0]
    for i in range(3,len(_prompt),1):
        if i !=len(_prompt)-2:            
            new_prompt+=_prompt[i]+"\n"
    
    return new_prompt
def prompt_template4(prompt):
    new_prompt=prompt.replace("Observation","What do you see").replace("You see","You have").replace("ball","beautiful ball").replace("step","stride").replace("key","beautiful key")
    return new_prompt
def prompt_template5(prompt,steps):
    new_prompt=""
    _prompt=prompt.split("\n")
    new_prompt+=_prompt[1]+"\n"+_prompt[0]+"\n"
    new_prompt+="\nyou have done "+str(steps)+" steps,"+str(64-steps)+"tries left\n"
    for i in range(2,len(_prompt),1):
        new_prompt+=_prompt[i]+"\n"
    return new_prompt
def prompt_template6(prompt):
    new_prompt=""
    _prompt=prompt.split("\n")
    new_prompt+=_prompt[1]+"\n"
    for i in range(2,len(_prompt),1):
        new_prompt+=_prompt[i]+"\n"
    return new_prompt
def prompt_template7(past_transitions, obs, info):
    prompt = "Possible actions of the agent: {}\n".format(", ".join(info["possible_actions"]))
    prompt += "{}\n".format(info["goal"])
    #prompt += "If you want to move forward, just say 'go' If you feel like turning right, you just need to say 'right' and if you want to turn left, you just need to say 'left' If you want to pick up an object in front of you, simply say 'pick' If you want to drop an object, say 'drop' And if you want to use an object to open a door, you just need to say 'toggle'.\n" 

    prompt += "Current Observation: {}\n".format(', '.join(obs))
    prompt += "Action:"
    return prompt
def prompt_template8(past_actions, obs, info):
    prompt = "Possible actions of the agent: {}\n".format(", ".join(info["possible_actions"]))
    prompt += "{}\n".format(info["goal"])
    #prompt += "If you want to move forward, just say 'go' If you feel like turning right, you just need to say 'right' and if you want to turn left, you just need to say 'left' If you want to pick up an object in front of you, simply say 'pick' If you want to drop an object, say 'drop' And if you want to use an object to open a door, you just need to say 'toggle'.\n" 
    prompt += "actions done: [{}}\n".format(', '.join(past_actions))
    prompt += "Current Observation: {}\n".format(', '.join(obs))
    prompt += "Action:"
    return prompt
def prompt_template9( obs, info):
    prompt=""
    prompt += "{}\n".format(info["goal"])
    prompt += "Current Observation: {}\n".format(', '.join(obs))
    prompt += "Action:"
    return prompt
def pick_a_prompt(past_transitions, obs, info,past_actions=[],prompt_number=0,steps=0):
    prompt=""
    if prompt_number==0:
        prompt=generate_prompt(past_transitions,obs,info)
        return prompt
    elif (prompt_number<=4 and prompt_number>0 ):
        _prompt_function=[prompt_template1,prompt_template2,prompt_template3,prompt_template4]
        prompt=_prompt_function[prompt_number-1](generate_prompt(past_transitions,obs,info))
        return prompt
    elif prompt_number==5:
        prompt=generate_prompt(past_transitions,obs,info)
        prompt=prompt_template5(prompt,steps)
        return prompt
    elif prompt_number==7:
        prompt=prompt_template7(past_transitions,obs,info)
        return prompt
    elif prompt_number==8:
        prompt=prompt_template8(past_actions,obs,info)
        return prompt
    elif prompt_number==9:
        prompt=prompt_template9(obs,info)
        return prompt
    else:
        print("not implemented")

