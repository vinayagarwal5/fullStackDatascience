import gradio as gr
from nltk.chat.util import Chat, reflections

# Your pairs configuration
pairs = [
    [r"(.*)my name is (.*)", ["Hello %2, How are you today ?",]],
    [r"(.*)help(.*) ", ["I can help you ",]],
    [r"(.*) your name ?", ["My name is thecleverprogrammer, but you can just call me robot and I'm a chatbot .",]],
    [r"how are you (.*) ?", ["I'm doing very well", "i am great !"]],
    [r"sorry (.*)", ["Its alright", "Its OK, never mind that",]],
    [r"i'm (.*) (good|well|okay|ok)", ["Nice to hear that", "Alright, great !",]],
    [r"(hi|hey|hello|hola|holla)(.*)", ["Hello", "Hey there",]],
    [r"what (.*) want ?", ["Make me an offer I can't refuse",]],
    [r"(.*)created(.*)", ["prakash created me using Python's NLTK library ", "top secret ;)",]],
    [r"(.*) (location|city) ?", ['hyderabad, India',]],
    [r"(.*)raining in (.*)", ["No rain in the past 4 days here in %2", "In %2 there is a 50% chance of rain",]],
    [r"how (.*) health (.*)", ["Health is very important, but I am a computer, so I don't need to worry about my health ",]],
    [r"(.*)(sports|game|sport)(.*)", ["I'm a very big fan of Cricket",]],
    [r"who (.*) (Cricketer|Batsman)?", ["Virat Kohli"]],
    [r"quit", ["Bye for now. See you soon :) ", "It was nice talking to you. See you soon :)"]],
    [r"(.*)", ['our customer service will reach you']]
]

# Initialize NLTK Chat
chat = Chat(pairs, reflections)

# Gradio predict function
def chatbot_response(message, history):
    # Get response from NLTK chat engine
    reply = chat.respond(message)
    if not reply:
        reply = "our customer service will reach you"
    return reply

# Create the Gradio interface
demo = gr.ChatInterface(
    fn=chatbot_response,
    title="NLTK Chatbot",
    description="Hi, I'm thecleverprogrammer and I like to chat. Please type lowercase English language to start.",
    examples=["hi", "what is your name?", "who is your favorite cricketer?"],
    # 'type' argument removed — newer gradio versions don't accept it here
)

if __name__ == "__main__":
    demo.launch()
