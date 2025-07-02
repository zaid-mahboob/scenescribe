import copy
import openai


conversation_history = [
    {
        "role": "system",
        "content": "This is the chat history between the user and the assistant. Use the conversation below as context when generating responses. Be concise and helpful.",
    }
]


def explanation_agent_1(image_base64, user_input):
    # Customize the prompt for Agent 1
    prompt = f"""
    “I am visually disabled. You are an
    assistant for individuals with visual disability. Your role is
    to provide helpful information and assistance based on my
    query. Your task is to {user_input}. Don’t mention that I
    am visually disabled or extra information to offend me. Be
    straightforward with me in communicating and don’t add any
    future required output, tell me what asked only
    """
    messages = {
        "role": "user",
        "content": [
            {"type": "text", "text": user_input},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ],
    }

    temp_history = copy.deepcopy(conversation_history)
    temp_history.append(messages)
    # print(temp_history)
    print("Content Prepared")
    # Call OpenAI API with image and text input, including conversation history
    completion = openai.chat.completions.create(
        model="gpt-4o-2024-05-13", messages=temp_history
    )

    # Get the AI's response content
    response_content = completion.choices[0].message.content

    # converter.say(response_content)
    # converter.runAndWait()
    # print(response_content)
    return response_content


def explanation_agent_2(user_input, agent_1_output):
    # Customize the prompt for Agent 2
    prompt = f"""
    I am visually disabled. You
    are an assistant for individuals with visual disability. Your
    role is to shrink the given information into a couple
    of lines in order to reduce the cognitive overloading. Your
    task is to remove all the unnecessary information from the
    given given information. Only keep information that is relevant
    to this query {user_input} Don’t mention that I am visually
    disabled to offend me, or that many details that he feels that
    he wishes he could see Avoid extra information like type kinds
    category so he felt disabled for not able to judge itself. since
    he’s blind so don’t start like this image or in the image and
    remove extra information that is not required to tell the blind.
    don’t add information by which I had to use my eyes and I
    feel disabled. Scene Description: {agent_1_output}.
    """
    messages = {"role": "user", "content": prompt}
    messages_ = {"role": "user", "content": user_input}
    print("Content Prepared")

    temp_history = copy.deepcopy(conversation_history)
    temp_history.append(messages)
    # Call OpenAI API with image and text input, including conversation history
    conversation_history.append(messages_)
    completion = openai.chat.completions.create(
        model="gpt-4o-mini", messages=temp_history
    )
    output = completion.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": output})
    # print(conversation_history)

    # converter.say(output)
    # converter.runAndWait()
    return output


def navigation_agent_1(image_base64, user_input):
    # Customize the prompt for Agent 1
    prompt = f"""I am visually disabled. You are an
    navigation assistant for individuals with visual disability. Your role is
    to provide navigation and direction assistance based on input
    query. Your task is to be a navigation assistant for the blind people, and provide navigation based answer to the query that is {user_input} with the help of image, don’t add any
    future required output, tell me what asked only. You have to guide me the user in terms of navigation telling in which direction should they move
    how many estimated steps are needed to reach the destination, if the destination is not so clear in the image, use your common sense,
    to judge how a human will use his/her brain with the given image to decide what should be the logical navigation and direction for reaching end goal.
    There is a possibility of some objects that might come in your way, if you see something in the image please add a small concise warning in the output without cognative overloading,
    don't give warnings if it's not referred from the image. Reminder that you need to navigate the pperson as per his requirements not to chit chat and don't use that you can't help, you are the only source
    """
    messages = {
        "role": "user",
        "content": [
            {"type": "text", "text": user_input},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ],
    }

    temp_history = copy.deepcopy(conversation_history)
    temp_history.append(messages)
    # print(temp_history)
    print("Content Prepared")
    # Call OpenAI API with image and text input, including conversation history
    completion = openai.chat.completions.create(
        model="gpt-4o-2024-05-13", messages=temp_history
    )

    # Get the AI's response content
    response_content = completion.choices[0].message.content

    # converter.say(response_content)
    # converter.runAndWait()
    # print(response_content)
    return response_content


def navigation_agent_2(user_input, agent_1_output):
    # Customize the prompt for Agent 2
    prompt = f"""
    I am visually disabled. You
    are an assistant for individuals with visual disability. Your
    role is to shrink the given information into a couple
    of lines in order to reduce the cognitive overloading. Your
    task is to remove all the unnecessary information from the
    given given information. Only keep information that is relevant
    to this query {user_input} Don’t mention that I am visually
    disabled to offend me, or that many details that he feels that
    he wishes he could see Avoid extra information like type kinds
    category so he felt disabled for not able to judge itself. since
    he’s blind so don’t start like this image or in the image and
    remove extra information that is not required to tell the blind.
    don’t add information by which I had to use my eyes and I
    feel disabled. Scene Description: {agent_1_output}.
    """
    messages = {"role": "user", "content": prompt}
    messages_ = {"role": "user", "content": user_input}
    print("Content Prepared")

    temp_history = copy.deepcopy(conversation_history)
    temp_history.append(messages)
    # Call OpenAI API with image and text input, including conversation history
    conversation_history.append(messages_)
    completion = openai.chat.completions.create(
        model="gpt-4o-mini", messages=temp_history
    )
    output = completion.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": output})
    # print(conversation_history)

    # converter.say(output)
    # converter.runAndWait()
    return output


def global_navigation_agent(user_input, tree):
    # Customize the prompt for Agent 1
    prompt = f"""
    “You are an assistant of visually impaired people, your task is to take user input and return only two things, one would be initial position and other
    final posiion. You are given user query and a hierarchical tree which represents a map of a building, you need to find the most optimum and logical position
    based on the user query and tree.

    You only have to give answer in a json format:

    {{
        "initial_position" = "",
        "final_position" = ""
    }}

    Tree is given below:
    {tree}

    User Query is this: {user_input}
    """
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt}]}]
    # print("Content Prepared")
    # Call OpenAI API with image and text input, including conversation history
    completion = openai.chat.completions.create(
        model="gpt-4o-mini", messages=messages)

    # Get the AI's response content
    response_content = completion.choices[0].message.content
    # converter.say(response_content)
    # converter.runAndWait()
    # print(response_content)
    return response_content
