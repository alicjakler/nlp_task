import spacy

# nlp = spacy.load('en_core_web_md')
nlp = spacy.load('en_core_web_sm')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))  # cat and monkey are quite similar
print(word3.similarity(word2))  # banana and monkey have some similarity
print(word3.similarity(word1))  # banana and cat have a low similarity

tokens = nlp("cat monkey banana ")

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
            "Hello, there is my car",
            "I\'ve lost my car in my car",
            "I\'d like my boat back",
            "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)


my_word1 = nlp("bicycle")
my_word2 = nlp("car")
my_word3 = nlp("helmet")

print(f"{my_word1} + {my_word2} = {my_word1.similarity(my_word2)}")  # bicycle and car are quite similar (both are modes of transport)
print(f"{my_word3} + {my_word2} = {my_word3.similarity(my_word2)}")  # helmet and car have some similarity (helmet for race cars)
print(f"{my_word3} + {my_word1} = {my_word3.similarity(my_word1)}")  # helmet and bicycle have more similarity (common for biker to wear helmet)

# using simple model 'en_core_web_sm' seems to report higher similarity that the 'en_core_web_md' though it didnt find much similarity between bicycle and helmet
# there is also a warning that the simple model has no word vectors loaded, so the similarity method will be based on components of the word rather than the words themselves
