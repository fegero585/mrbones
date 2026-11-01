import ears

print("Say 'Hey Mister Bones'...")
ears.wait_for_wake_word()
print("Wake word heard! Now ask a question...")

question = ears.listen_for_question()
if question:
    print(f"I heard: {question}")
else:
    print("I didn't hear anything.")
