""""Module: gpt_tests.py
This is a file contating functions playing around with the functionality of the gpt 3.5
Python API."""
import openai


def init_api():
    """This function initializes the openai api with the key stored in the .env file."""
    openai.my_api_key = ''


if __name__ == '__main__':
    openai.api_key = ''
    messages = [ {"role": "system", "content": "You are a intelligent assistant."} ]

    while True:
        message = input("Enter a message: ")
        if message:
            messages.append(
                {"role": "user", "content": message},
            )
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )
        
        reply = chat.choices[0].message.content
        print(f"ChatGPT: {reply}")
        messages.append({"role": "assistant", "content": reply})