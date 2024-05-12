import gradio as gr
import time
from arxiv_agent import *
agent = ArxivAgent()

def set_profile(name):
    # Simulate setting the profile based on the name
    # Replace with actual logic to fetch and set profile
    names = name.split(" ")
    for n in names:
        if len(n) == 0: 
            gr.Info("Please input standard name format.")
            return None
        elif n[0].islower(): 
            gr.Info("Please input standard name format.")
            return None
    profile = agent.get_profile(name)
    # import pdb
    # pdb.set_trace()

    return profile


def edit_profile(profile, author_name):
    # names = author_name.split(" ")
    # for n in names:
    #     if len(n) == 0: 
    #         gr.Info("Please input standard name format.")
    #         return "", False
    #     elif n[0].islower(): 
    #         gr.Info("Please input standard name format.")
    #         return "", False

    msg = agent.edit_profile(profile, author_name)
    gr.Info("Edit profile successfully!")
    return profile

def confirm_date(date, profile_input):
    # Simulate fetching data based on the selected date
    # data = request.get_json()
    if len(profile_input) == 0:
        topic, papers, idea = agent.select_date(date, None)
    else:
        topic, papers, idea = agent.select_date(date, profile_input)
    return topic[0], papers, idea[0]

def send_text(query, profile_input):
    # Simulate sending a query and receiving a response
    if len(profile_input) <= 1:
        Ans1, Ans2 = agent.response(query, None)
    else:
        Ans1, Ans2 = agent.response(query, profile_input)

    return Ans1[0], Ans2[0]



def send_comment(comment):
    # Simulate sending a comment
    message = agent.update_comment(comment)
    gr.Info("Thank you for your comment!")

    return message[0]



def respond(message, chat_history, profile):
    
    bot_message1, bot_message2 = send_text(message, profile)


    # bot_message1, bot_message2 = "a", "b"
    chat_history.append((message, None))
    chat_history.append((bot_message1, bot_message2))

    time.sleep(2)

    return "", chat_history

  

with gr.Blocks(css="""#chat_container {height: 820px; width: 1000px; margin-left: auto; margin-right: auto;}
            #chatbot {height: 600px; overflow: auto;}
            #create_container {height: 750px; margin-left: 0px; margin-right: 0px;}
            #tokenizer_renderer span {white-space: pre-wrap}
            """,
    theme="bethecloud/storj_theme",title="Arxiv Copilot") as app:
    with gr.Row():
        with gr.Column(scale=2):
            gr.Image(
                "images/arxiv_copilot.PNG", elem_id="banner-image", show_label=False
            )
        with gr.Column(scale=5):
            gr.Markdown(
                """# Arxiv Copilot
                ➡️️ **Goals**: Arxiv Copilot aims to provide personalized academic service! 
                
                ✨ **Guidance**: 
                
                Step (1) Enter researcher name and generate research profile in "Set your profile!"🧑‍💼
                
                Step (2) Select time range and get relevant topic trend and ideas in "Get research trend and ideas!"💡

                Step (3) Chat with Arxiv Copilot and choose the better response from two answers in "Chat with Arxiv Copilot!"; Here we appreciate any further feedback 🎉
                
                ⚠️ **Limitations**: We mainly provide research service related to machine learning field now, other fields will be added in the future.

                🗄️ **Disclaimer**: User behavior data will be collected for the pure research purpose. If you use this demo, you may implicitly agree to these terms.
                """
            )


    # gr.Markdown("Provide research service using this demo.")
    with gr.Accordion("Set your profile!", open=True):
        gr.Markdown(
            """
            You can input your name in standard format to get your profile from arxiv here. Standard examples: Yoshua Bengio. Wrong examples: yoshua bengio, Yoshua bengio, yoshua Bengio.
            """
        )
        with gr.Row():
            with gr.Column(scale=2, min_width=300):
                name_input = gr.Textbox(label="Input Your Name")
                set_button = gr.Button("Set Profile")
            profile_text = gr.Textbox(label="Generated Profile", interactive=True, scale=7, lines=5, max_lines=5)
            edit_button = gr.Button("Edit Profile", scale=1)
        set_button.click(set_profile, inputs=name_input, outputs=[profile_text])
        edit_button.click(edit_profile, inputs=[profile_text, name_input], outputs=[profile_text])

    with gr.Accordion("Get research trend and ideas!", open=True):
        gr.Markdown(
            """
            We will give you personalized research trend and ideas if you have set your profile. Otherwise, general research trend will be provided.
            """
        )
        with gr.Column():
            with gr.Row():
                with gr.Column(scale=2, min_width=300):
                    # gr.Dropdown(
                    #     ["day", "week", "bird"], label="Select time range", info="Will add more animals later!"
                    # ),
                    date_choice = gr.Radio(["day", "week", "all"], label="Select Time Range", value="day")
                    date_button = gr.Button("Confirm")
                papers_text = gr.Textbox(label="Trend Papers", interactive=False, scale=8, lines=5, max_lines=5)

            with gr.Row():
                topic_text = gr.Textbox(label="Topic Trend", interactive=False, scale=5, lines=7, max_lines=10)

                ideas_text = gr.Textbox(label="Ideas Related to Topic Trend", interactive=False, scale=5, lines=7, max_lines=10)

        date_button.click(confirm_date, inputs=[date_choice, profile_text], outputs=[topic_text, papers_text, ideas_text])

    with gr.Accordion("Chat with Arxiv Copilot!", open=True):
        gr.Markdown(
            """
            Each time we will give you two answers. If you prefer the second answer, you can click 👍 below the second answer and the first answer will be removed. If you click 👎, the second answer will be removed. 
            """
        )
        with gr.Column():
            chatbot = gr.Chatbot()
            with gr.Row():
                msg = gr.Textbox(placeholder="Message Arxiv Copilot here...", scale=9, show_label=False)
                send_button = gr.Button("Send",scale=1)  # Adding a Send button
                clear = gr.ClearButton([msg, chatbot],scale=1)

            


        def print_like_dislike(x: gr.LikeData, chat_history):
            cur_index = x.index[0]
            if  cur_index >= 1 and chat_history[cur_index - 1][1] is None:
                if x.liked:
                    chat_history[cur_index - 1][1] = chat_history[cur_index][1]
                    agent.update_feedback_thought(chat_history[cur_index - 1][0], chat_history[cur_index][0], chat_history[cur_index][1], 0, 1)
                    # gr.Info("You like the second answer, and the fisrt answer will be removed.")

                else:
                    agent.update_feedback_thought(chat_history[cur_index - 1][0], chat_history[cur_index][0], chat_history[cur_index][1], 1, 0)
                    chat_history[cur_index - 1][1] = chat_history[cur_index][0]
                    # gr.Info("You dislike the second answer, and the second answer will be removed.")
                chat_history.remove(chat_history[cur_index])
            else:
                gr.Info("You have gave your feedback. You can ask more questions.")
            return chat_history



        
        msg.submit(respond, [msg, chatbot, profile_text], [msg, chatbot])    # Set up the action for the Send button
        send_button.click(respond, inputs=[msg, chatbot, profile_text], outputs=[msg, chatbot])
        chatbot.like(print_like_dislike, [chatbot], [chatbot])


        with gr.Row():
            comment_input = gr.Textbox(label="With Arxiv Copilot, how much time do you save to obtain the same amount of information?", scale=9, lines=3)
            comment_button = gr.Button(value="Comment", scale=1)


        comment_button.click(send_comment, inputs=comment_input, outputs=None)

    

app.launch()
