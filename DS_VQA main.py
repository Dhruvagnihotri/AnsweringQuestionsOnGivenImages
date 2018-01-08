#Visual Question Answering - D-VQA Technique
#By Suraj Kiran Raman, VijayaKrishna Naganoor, Shanthakumar Venkatraman, Dhruv Agnihotri

# In developing the DVQA model, we use the computational packages from theano with 
# the Keras wrapper and the kraino library developed by Malinowski et al. for VQA.
import numpy as np
import sys
import theano
import keras

################# Loading Dataset #################

# The question answer pairs are given in the form of a triple containing questions and answers and the corresponding image indices.
# Loading the DAQUAR question-answer pairs.
from kraino.utils import data_provider
Data_Triples = data_provider.select['daquar-triples']

test_question_set = Data_Triples['text'](train_or_test='test')
train_question_set = dp['text'](train_or_test='train')

train_questions = train_question_set['x']#Extracting the questions component.
train_answers = train_question_set['y']#Extracting the answers component.
test_questions = test_question_set['x']
test_answers= test_question_set['y']

train_image_names = train_question_set['img_name']
test_image_names = test_text_representation['img_name']

################# 				 #################


################# Encoding Train Questions #################

# Convert questions and answers into One-Hot vectors
# Here we define one-hot vector indices in the order of descending frequencies of the words
word2index_x = Estimate_Frequency(train_questions)
# Kraino library built for VQA models specifically provides us a method to encode these as one hot vectors.
from kraino.utils.input_output_space import encode_questions_index
train_ques_one_hot = encode_questions_index(train_questions, word2index_x)
# Questions are of variable length, so we pad zeros at the beginning of each question to 
# define the question length as a constant 30. This can be efficiently done using Keras
from keras.preprocessing import sequence
MAXLEN=30
train_encoded_ques = sequence.pad_sequences(train_ques_one_hot, maxlen=MAXLEN)
MAX_ANSWER_TIME_STEPS=1

################# 				 		    #################

################# Encoding Train Answers #################

# Similar to how we have encoded questions, we encode answers. Restrict to single word answers
word2index_y = Estimate_Frequency(train_answers)
from kraino.utils.input_output_space import encode_answers_one_hot
word2index_y, index2word_y = build_vocabulary(this_wordcount=wordcount_y)
train_encoded_ans, _ = encode_answers_one_hot(
train_answers, 
word2index_y, 
answer_words_delimiter=train_text_representation['answer_words_delimiter'],
is_only_first_answer_word=True,
max_answer_time_steps=MAX_ANSWER_TIME_STEPS)

################# 				 		    #################

################# Encoding Test Questions #################

test_ques_one_hot = encode_questions_index(test_questions, word2index_x)
test_encoded_ques = sequence.pad_sequences(test_ques_one_hot, maxlen=MAXLEN)

################# 				 		    #################

################ Loading our proposed DVQA model feature ################

# Concat_Feature_NPY.npy here contains the extracted features for our DVQA model using MATLAB. Refer to Instructions text file on how to generate it.
train_visual_features_DVQA=np.load('Concat_Feature_NPY.npy');
train_input = [train_encoded_ques, train_visual_features_DVQA];

##################          					         #################

############### Defining the RNN-LSTM model #############
# We can efficientlty define the model using the keras and kraino libraries
# First we define a model using keras/kraino

# Defining hyperparameters of LSTM
EMBEDDING_DIM = 500
# Here we have experimented and validated our performance on multiple fusion modalities.
MULTIMODAL_MERGE_MODE = 'sum'

model_config = Config(
textual_embedding_dim=EMBEDDING_DIM,
visual_embedding_dim=EMBEDDING_DIM,
hidden_state_dim=EMBEDDING_DIM,
multimodal_merge_mode=MULTIMODAL_MERGE_MODE,
input_dim=len(word2index_x.keys()),
output_dim=len(word2index_y.keys()),
visual_dim=train_visual_features_DVQA.shape[1])

Model_Definition(model_config)
model.create()

# here we have experimented and validated our performance on multiple optimizers.
model.compile(
loss='categorical_crossentropy', 
optimizer='Adam')
text_image_rnn_model_DVQA = model

############### 							 #############


############### Training the DVQA LSTM model #############

#== Model training
text_image_rnn_model_DVQA.fit(
train_input, 
train_encoded_ans,
batch_size=512,
nb_epoch=30,
validation_split=0.1,
show_accuracy=True)

############### 							 #############

# Now we have to define the switching model 

################# Switching Model ###############
# We used pretrained Res-Net model for evaluating the switchinig model in DS-VQA

 CNN_NAME='fb_resnet'
 PERCEPTION_LAYER='l2_res5c-152' # l2 prefix since there are l2-normalized visual features

 train_visual_features_Normal = Data_Triples['perception'](
 train_or_test='train',
 names_list=train_image_names,
 parts_extractor=None,
 max_parts=None,
 perception=CNN_NAME,
 layer=PERCEPTION_LAYER,
 second_layer=None
 )
################                      ##################

############## Defining the RNN-LSTM model for the switching #############
# We can efficientlty define the model using the keras and kraino libraries
# First we define a model using keras/kraino

# Defining hyperparameters of LSTM
EMBEDDING_DIM = 500
# Here we have experimented and validated our performance on multiple fusion modalities.
MULTIMODAL_MERGE_MODE = 'sum'

model_config = Config(
textual_embedding_dim=EMBEDDING_DIM,
visual_embedding_dim=EMBEDDING_DIM,
hidden_state_dim=EMBEDDING_DIM,
multimodal_merge_mode=MULTIMODAL_MERGE_MODE,
input_dim=len(word2index_x.keys()),
output_dim=len(word2index_y.keys()),
visual_dim=train_visual_features_Normal.shape[1])  # Important: Note that here we are passing the Switching model's visual features.

Model_Definition(model_config)
model_switch.create()

# here we have experimented and validated our performance on multiple optimizers.
model_switch.compile(
loss='categorical_crossentropy', 
optimizer='Adam')
text_image_rnn_model_switch = model

############### 							 #############

############### Training the DSVQA - Switch LSTM model #############

#== Model training
text_image_rnn_model_switch.fit(
train_input, 
train_encoded_ans,
batch_size=512,
nb_epoch=30,
validation_split=0.1,
show_accuracy=True)

############### 							 #############

# Now we have our model definied. We can now predict the answers and evaluate accuracies.

############## Our Proposed DS-VQA model Switching Part ############
# For testing our DSVQA, we load the extracted features (on MATLAB) refer the instructions document.
# Parse the question for depth keywords and develop a switch from DVQA and "Resnet152+LSTM" model
test_visual_features_DVQA=np.load('Test_Features_Concat.npy');
test_input = [test_encoded_ques, test_visual_features];
import copy;
size=1000;
Final_predictions=[];
test_x_temp=copy.deepcopy(test_encoded_ques);
test_visual_features_temp=copy.deepcopy(test_visual_features_DVQA);
test_visual_features_normal_temp=copy.deepcopy(train_visual_features_Normal);
count1=0;
count2=0;
for i in range(0,size):
    ques=test_raw_x[i];
    Ques_words=ques.split()
    test_x_temp[0:size]=0;
    test_x_temp[i]=test_encoded_ques[i];
    print i;
    test_visual_features_temp[0:size]=0;
    test_visual_features_normal_temp[0:size]=0;
    if "behind" in Ques_words or "front" in Ques_words or "far" in Ques_words or "close" in Ques_words:  # Depth Vocabulary
        count1=count1+1;
        test_visual_features_temp[i]=test_visual_features_DVQA[i];
        test_input = [test_x_temp, test_visual_features_temp];
        from kraino.core.model_zoo import word_generator
        text_image_rnn_model._config.word_generator = word_generator['max_likelihood']
        predictions_answers = text_image_rnn_model.decode_predictions(
        X=test_input,
        temperature=None,
        index2word=index2word_y,
        verbose=0)
    else:
        count2=count2+1;
        test_visual_features_normal_temp[i]=train_visual_features_Normal[i];
        test_input = [test_x_temp, test_visual_features_normal_temp];
        from kraino.core.model_zoo import word_generator
        text_image_rnn_model_normal._config.word_generator = word_generator['max_likelihood']
        predictions_answers = text_image_rnn_model_normal.decode_predictions(
        X=test_input,
        temperature=None,
        index2word=index2word_y,
        verbose=0)
    Final_predictions.append(predictions_answers[i])


###################### 													#########################

# Link the standard NTLK Data to load the similarity dictionary and compute WUPS score.
%env NLTK_DATA=C:\Users\Dell user\Downloads\visual_turing_test-tutorial\visual_turing_test-tutorial\data\nltk_data

#################  Compute WUPS Score #################
from kraino.utils import print_metrics
_ = print_metrics.select['wups'](
gt_list=test_raw_answers,
pred_list=Final_predictions,
verbose=1,
extra_vars=None)
################# 						##################