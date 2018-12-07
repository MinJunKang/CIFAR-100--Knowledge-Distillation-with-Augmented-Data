

import Header
from PIL import Image
from PIL import ImageFile
import itertools

# No Dropout, Simple DNN Structure
class Student_model(object):
    def __init__(self, args, model_name,label_num):
        # Flag info
        self.flag = False

        self.model_name = model_name

        # Define Argument
        self.Define_Arg(args)

        # Training Parameters
        self.num_parameter = 0

        # To determine if it is trainable
        self.trainable = False

        self.last_epoch = 0
        self.input_shape =[224,224,3]
        self.output_shape = [label_num]
        self.num_class = label_num


    def Define_Arg(self,args):
        # Temperature
        self.temperature = args.temperature

        # Learning parameter
        self.min_learning_rate = args.min_learning_rate
        self.learning_rate_increment = args.learning_rate_increment
        self.max_learning_rate = args.max_learning_rate
        self.min_batch = args.min_batch
        self.batch_increment = args.batch_increment
        self.max_batch = args.max_batch

        # Structure parameter
        self.min_layer = args.min_layer
        self.layer_increment = args.layer_increment
        self.max_layer = args.max_layer

        self.min_node = args.min_node
        self.node_increment = args.node_increment
        self.max_node = args.max_node
        
        # Set epoch and stop point criterion
        self.max_epoch = args.max_epoch
        self.max_overfit = args.max_overfit

        # model type
        self.model_type = args.model_type

        # Saving place
        self.dir = args.dir
        self.checkpoint_dir = args.checkpoint
        self.final_dir = args.final
        self.log_dir =args.log
        self.model_dir = args.model
        self.tmp_dir = args.tmp

        self.checkpoint_file = Header.os.path.join(self.checkpoint_dir,"checkpoint_weight.h5")
        self.final_file = Header.os.path.join(self.final_dir,"trained_weight.h5")
        self.log_file_1 = Header.os.path.join(self.log_dir,"Result_model.txt")
        self.model_file = Header.os.path.join(self.model_dir,"model.json")
        self.tmp_file_1 = Header.os.path.join(self.tmp_dir,"model.json")
        self.tmp_file_2 = Header.os.path.join(self.tmp_dir,"trained_weight.h5")
        self.tmp_file_3 = Header.os.path.join(self.tmp_dir,"info.txt")

    # Initialize the weight and bias
    def get_stddev(self,in_dim, out_dim):
        return 1.3 / Header.math.sqrt(float(in_dim) + float(out_dim))

    # Loss function
    def loss(self,y_true,y_pred):
        if(self.model_type == "classifier"):
            pred = Header.K.softmax(y_pred)

            total_loss = Header.tf.reduce_mean(Header.K.categorical_crossentropy(y_true,pred))

            loss_op_soft = Header.tf.cond(Header.tf.constant(self.flag),
                                        true_fn=lambda: Header.tf.reduce_mean(Header.K.categorical_crossentropy(
                                           self.soft_true, Header.K.softmax(y_pred / self.temperature))),
                                        false_fn=lambda: 0.0)
        else:
            pred = Header.K.tanh(y_pred)

            total_loss = Header.tf.reduce_mean(Header.tf.square(pred - y_true))

            loss_op_soft = Header.tf.cond(Header.tf.constant(self.flag),
                                        true_fn=lambda: Header.tf.reduce_mean(Header.tf.square(Header.K.tanh(y_pred/ self.temperature) - self.soft_true)),
                                        false_fn=lambda: 0.0)

        total_loss += Header.tf.square(self.temperature) * loss_op_soft

        return total_loss

    def loss_train_student(self,y_true,y_pred):  ##loss 留욌뒗吏  ?뺤씤 
        if(self.model_type == "classifier"):
            pred = Header.K.softmax(y_pred)

            total_loss = Header.tf.reduce_mean(Header.K.categorical_crossentropy(y_true,pred))
        else:
            pred = Header.K.tanh(y_pred)

            total_loss = Header.tf.reduce_mean(Header.tf.square(pred - y_true))
        return total_loss
    
    def acc_model(self,y_true,y_pred):
        if(self.model_type == "classifier"):
            prediction = Header.K.softmax(y_pred)
            accuracy = Header.tf.reduce_mean(Header.tf.cast(Header.tf.equal(Header.tf.argmax(prediction,1),Header.tf.argmax(y_true,1)),"float32"),name = "Accuracy_classifier")
        else:
            accuracy = Header.tf.reduce_mean(Header.tf.square(y_pred - y_true,name = "squared_accuracy"),name = "Accuracy_regression")
        return accuracy

    def Load_Text(self,dst_file):
        contents = []
        with open(dst_file,'rb') as myloaddata:
            contents = Header.pickle.load(myloaddata)
        return contents

    def Build_DNN(self,hidden_units=[256,256]):

        soft_y = Header.l.Input(shape=self.output_shape,name="%s_soft_y" % self.model_name)
        self.soft_true = soft_y

        inputs = Header.l.Input(shape = self.input_shape)
        conv_1 = Header.l.Conv2D(filters = 16,kernel_size = (3,3),strides = (2,2))(inputs)
        act_1 = Header.l.Activation('relu')(conv_1)
        conv_2 = Header.l.Conv2D(filters = 16,kernel_size = (3,3),strides = (2,2))(act_1)
        act_2 = Header.l.Activation('relu')(conv_2)
        conv_3 = Header.l.Conv2D(filters = 16,kernel_size = (3,3),strides = (2,2))(act_2)
        batch_3 = Header.l.BatchNormalization()(conv_3)
        act_3 = Header.l.Activation('relu')(batch_3)
        conv_4 = Header.l.Conv2D(filters = 16,kernel_size = (5,5),strides = (2,2))(act_3)
        batch_4 = Header.l.BatchNormalization()(conv_4)
        act_4 = Header.l.Activation('relu')(batch_4)
        pool_1 = Header.l.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(act_4)

        flat_1 = Header.l.Flatten()(pool_1)
        out_1 = Header.l.Dense(units = 2048,activation = 'relu')(flat_1)

        def Reshape(x):
            return Header.tf.reshape(x,shape=(Header.tf.shape(x)[0],128,16))

        def flip(x,axis):
            return Header.tf.reverse(x,[axis])

        layer1 = Header.l.Lambda(Reshape)(out_1)
        layer2 = Header.l.Lambda(flip,arguments={'axis':2})(layer1)
        layer3 = Header.l.Lambda(flip,arguments={'axis':1})(layer1)
        layer4 = Header.l.Lambda(flip,arguments={'axis':0})(layer1)

        lstm_lay1 = Header.l.LSTM(128,activation='relu',name = 'lstm_down')(layer1)
        lstm_lay2 = Header.l.LSTM(128,activation='relu',name = 'lstm_up')(layer2)
        lstm_lay3 = Header.l.LSTM(128,activation='relu',name = 'lstm_left')(layer3)
        lstm_lay4 = Header.l.LSTM(128,activation='relu',name = 'lstm_right')(layer4)

        lay_concat = Header.l.concatenate([lstm_lay1,lstm_lay2,lstm_lay3,lstm_lay4],1)

        out_ = Header.l.Dense(64)(lay_concat)
        out = Header.l.Dense(self.num_class)(out_)
        model = Header.m.Model([inputs,soft_y],out)

        #model = Header.m.Sequential(name = "%s_DNN_Model" % self.model_name)

        ## Hidden Layer 1

        #model.add(Header.l.Flatten(input_shape = self.input_shape))
        #model.add(Header.l.Dense(hidden_units[0],activation = "relu",kernel_initializer=Header.lz.TruncatedNormal(mean=0.0, stddev=self.get_stddev(self.in_dim,hidden_units[0]), seed=Header.seed),name = "Hidden_1",input_shape = [self.in_dim]))

        ## Hidden Layer 2 ~ N
        #for i in range(1,len(hidden_units)):
        #    model.add(Header.l.Dense(hidden_units[i],activation = "relu",kernel_initializer=Header.lz.TruncatedNormal(mean=0.0, stddev=self.get_stddev(hidden_units[i-1],hidden_units[i]), seed=Header.seed),name = "Hidden_%s" % str(i+1)))
        
        ## Last Output Layer
        #model.add(Header.l.Dense(self.num_class,activation="sigmoid",kernel_initializer=Header.lz.TruncatedNormal(mean=0.0, stddev=self.get_stddev(hidden_units[-1],self.out_dim), seed=Header.seed),name = "Output"))
        
        #model = Header.m.Model([model.input,soft_y],model.output)

        return model

    def Train(self,args,train_dir,valid_dir,train_num,valid_num,teacher_flag = False): ## teahcer_flag ?대쫫 蹂 寃?
     
        # Define Argument
        self.Define_Arg(args)
        soft_file = args.softy_file

        self.in_dim = 299*299*3
        self.out_dim = 50

        f = open(self.log_file_1,'w')
        f.write("Temperature : %f\n" % self.temperature)
        f.close()
        
        # Change the structure
        for layer_num in range(self.min_layer,self.max_layer+1):
            hidden_node = []
            for layer in range(layer_num):
                hidden_node.append(self.min_node)
            for layer in range(layer_num):
                if(layer != 0):
                    hidden_node[layer] *= self.node_increment
                while(hidden_node[layer] <= self.max_node):
                    f = open(self.log_file_1,'a')
                    result=self.Optimize(hidden_node,teacher_flag,train_dir,valid_dir,soft_file,train_num,valid_num)
                    f.write("-------------------------------\n")
                    f.write("Model : [")
                    for i in range(len(hidden_node)):
                        f.write("%d  "%hidden_node[i])
                    f.write("]\nTraining_info : [")
                    f.write("learning_rate : %f     "%result[0])
                    f.write("batch_size : %d    " % result[1])
                    f.write("Last epoch : %d]\n" % self.last_epoch)
                    f.write("Train accuracy : %f        Validation accuracy : %f\n" % (result[3],result[2]))
                    f.write("Number of parameter : %d\n" % self.num_parameter)
                    f.write("-------------------------------\n")
                    f.close()
                    hidden_node[layer] *= self.node_increment
                    if(result[2] == 1.00):
                        print("Found the best model!! Finish the training")
                        return
                hidden_node[layer] = int(hidden_node[layer] / self.node_increment)


    def Optimize(self,hidden_node,teacher_flag,train_dir,valid_dir,soft_file,train_num,valid_num):
        rate = self.min_learning_rate
        result = 0
        self.batch = self.min_batch
        data = []
        data.append(rate)
        data.append(self.min_batch)
        data.append(result) # Test accuracy
        data.append(result) # Train accuracy

        [train_result,tmp_result] = self.Run(hidden_node,rate,self.batch,result,teacher_flag,train_dir,valid_dir,soft_file,train_num,valid_num)
        if(result < tmp_result):
            data[0] = rate
            data[1] = self.min_batch
            data[2] = tmp_result
            data[3] = train_result
            result = tmp_result
            if(result == 1.00):
                return data

        return data

    # Still Programming
    def Run(self,hidden_node,learning_rate,batch_size,result,teacher_flag,train_dir,valid_dir,soft_file,train_num,valid_num):

        # Build Model
        model = self.Build_DNN(hidden_node)

        test_acc_save = 0
        last_epoch = 0

        # compile model
        self.adam = Header.op.Adam(lr=learning_rate,clipvalue=1.5)

        datagen = Header.ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
        

        if teacher_flag is False:
            datagen_num = 3
            train_gen = self.generate_arrays_from_file(train_dir,None,batch_size,datagen_num,datagen)
            valid_gen = self.generate_arrays_from_file(valid_dir,None,batch_size,datagen_num,datagen)
        else:
            if(Header.os.path.isfile(self.model_file) == False):
                print("Train the plain student model first!!")
                exit(1)
            loaded_model_json = Header.m.load_model(self.model_file,custom_objects = {'loss':self.loss,'acc_model':self.acc_model})
            model = Header.m.model_from_json(loaded_model_json.to_json())
            model.load_weights(self.final_file)
            datagen_num = 0
            train_gen = self.generate_arrays_from_file(train_dir,soft_file,batch_size,datagen_num,None)
            valid_gen = self.generate_arrays_from_file(valid_dir,soft_file,batch_size,datagen_num,None)

        train_data_count = train_num
        valid_data_count = valid_num

        train_per_epoch = round(train_data_count * (datagen_num + 1) / batch_size + 0.5)
        valid_per_epoch = round(valid_data_count * (datagen_num + 1) / batch_size + 0.5)
        
        if teacher_flag: ## teacher_flag ?쒖슜
            model.compile(self.adam,loss=self.loss,metrics=[self.acc_model])
            self.flag = True 
        else:
            model.compile(self.adam,loss=self.loss_train_student,metrics=[self.acc_model])
            self.flag = False

        # Model information
        model.summary()
        self.num_parameter = int(Header.np.sum([Header.K.count_params(p) for p in set(model.trainable_weights)]))

        checkpointer = Header.call.ModelCheckpoint(filepath=self.checkpoint_file,monitor = "val_acc_model", verbose=1, save_best_only=True, save_weights_only=True)
        earlyStopping=Header.call.EarlyStopping(monitor='val_acc_model', patience=self.max_overfit, verbose=0, mode='auto')

        history = model.fit_generator(train_gen,steps_per_epoch = train_per_epoch,
                                      epochs = self.max_epoch, verbose = 2,callbacks=[checkpointer,earlyStopping],
                                      validation_data = valid_gen, validation_steps = valid_per_epoch,
                                      max_queue_size =100 , shuffle = False, initial_epoch = 0)
        # Get the value
        test_acc_save = Header.np.max(history.history['val_acc_model'])
        last_epoch = Header.np.argmax(history.history['val_acc_model'])+1
        test_loss_save = history.history['val_loss'][last_epoch-1]
        train_acc_save = history.history['acc_model'][last_epoch-1]
        train_loss_save = history.history['loss'][last_epoch-1]
        
        print("\n\nTrained Model Structure : ")
        print("Structure : ",hidden_node, "     Number of parameter : ",self.num_parameter)
        print("Train_parameter = learning_rate : ",learning_rate,"    batch_size : ",batch_size)
        print("Last Epoch : %d" % last_epoch)
        print("Validation Accuracy : ",test_acc_save,"      Validation loss : ",test_loss_save)
        print("Trainset Accuracy : ",train_acc_save,"       Trainset loss : ",train_loss_save)

        if(result < test_acc_save):
            self.last_epoch = last_epoch
            model.save(self.model_file)
            Header.shutil.copy(self.checkpoint_file,self.final_file)

        # return the result
        return [train_acc_save,test_acc_save]
   
    def Evaluate(self,test_dir,soft_file,test_num):
        if(Header.os.path.isfile(self.model_file) == False):
            print("Train the plain student model first!!")
            exit(1)

        loaded_model_json = Header.m.load_model(self.model_file,custom_objects = {'loss':self.loss,'acc_model':self.acc_model})

        batch_size = self.min_batch

         # Load Model
        loaded_model = Header.m.model_from_json(loaded_model_json.to_json())
        loaded_model.load_weights(self.final_file)

        loaded_model.compile(loss = self.loss, optimizer = self.adam,metrics = [self.acc_model])

        test_gen =  self.generate_arrays_from_file_valid(test_dir,soft_file,batch_size,0,None)
        test_data_count = test_num
        test_per_epoch = round(test_data_count / batch_size + 0.5)
        
        return loaded_model.evaluate_generator(test_gen,test_per_epoch)
    
    def Num_Parameter(self):
        if(self.num_parameter != 0):
            return self.num_parameter

    def Fit_SingleModel(self,train_dir,valid_dir,test_dir,soft_file,train_num,valid_num,test_num,hidden_node = [256,256],learn_rate = 0.001,batch = 128):
        

        # Build Model
        model = self.Build_DNN(hidden_node)

        model.compile(self.adam,loss=self.loss,metrics=[self.acc_model])

        batch_size = self.min_batch
       
        train_gen = self.generate_arrays_from_file(train_dir,soft_file,batch_size,0,None)
        valid_gen = self.generate_arrays_from_file(valid_dir,soft_file,batch_size,0,None)
        test_gen =  self.generate_arrays_from_file(test_dir,soft_file,batch_size,0,None)

        train_data_count = train_num
        valid_data_count = valid_num
        test_data_count = test_num

        train_per_epoch = round(train_data_count / batch_size + 0.5)
        valid_per_epoch = round(valid_data_count / batch_size + 0.5)
        test_per_epoch = round(test_data_count / batch_size + 0.5)

        # Model information
        model.summary()
        parameter_num = int(Header.np.sum([Header.K.count_params(p) for p in set(model.trainable_weights)]))
        self.num_parameter = parameter_num

        checkpointer = Header.call.ModelCheckpoint(filepath=self.tmp_file_2,monitor = "val_acc_model", verbose=1, save_best_only=True, save_weights_only=True)
        earlyStopping=Header.call.EarlyStopping(monitor='val_acc_model', patience=self.max_overfit, verbose=0, mode='auto')

        history = model.fit_generator(train_gen,steps_per_epoch = train_per_epoch,
                                      epochs = self.max_epoch, verbose = 2,callbacks=[checkpointer,earlyStopping],
                                      validation_data = valid_gen, validation_steps = valid_per_epoch,
                                      max_queue_size =100 , shuffle = True, initial_epoch = 0)

        # Get the value
        valid_acc_save = Header.np.max(history.history['val_acc_model'])
        last_epoch = Header.np.argmax(history.history['val_acc_model'])+1
        valid_loss_save = history.history['val_loss'][last_epoch-1]
        train_acc_save = history.history['acc_model'][last_epoch-1]
        train_loss_save = history.history['loss'][last_epoch-1]

        # Save model
        model.save(self.tmp_file_1)

        # Inference
        result = model.evaluate_generator(test_gen,test_per_epoch,verbose=3)
        test_acc_save = result[1]

        # Result view
        model.summary()

        # Save the result to txt file
        with open(self.tmp_file_3,'wt') as f:
            f.write("Temperature : %f\n" % self.temperature)
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write("----------------------------------------\n")
            f.write("Training Parameter Information : \n")
            f.write("learning_rate : %f     batch_size : %d\n" % (learn_rate,self.min_batch))
            f.write("Last Epoch : %d\n" % last_epoch)
            f.write("----------------------------------------\n")
            f.write("Train Accuracy : %f    Train loss : %f\n" % (train_acc_save,train_loss_save))
            f.write("Valid Accuracy : %f    Test Accuracy : %f\n" % (valid_acc_save,test_acc_save))
            f.write("----------------------------------------\n")
        return [test_acc_save,parameter_num]

    def Temp_file3_add(self,data):
        with open(self.tmp_file_3,'at') as f:
            f.write("Complexity_gain(Teacher params / Student params) : %f\n" % data[0])
            f.write("Accuracy_gain(Teacher acc - Student acc) : %f\n" % data[1])
            f.write("Overall_gain(Complexity_gain / Accuracy_gain) : %f\n" % data[2])

    def View_Temp_log(self):
        if(Header.os.path.isfile(self.tmp_file_3) == False):
            print("You must first train the temporary student model to get spec")
            return False
        else:
            with open(self.tmp_file_3) as f:
                for line in f:
                    print(line)
            return True

    def View_Train_log(self):
        if(Header.os.path.isfile(self.log_file_1) == False):
            print("You must first train the student model to get train log")
            return False
        else:
            with open(self.log_file_1) as f:
                for line in f:
                    print(line)
            return True
    def generate_arrays_from_file(self,feature_dir,soft_file,batch_size,data_gen_num,gen = None):

        length = len(Header.os.listdir(feature_dir))

        if soft_file is not None:
            with open(soft_file,'rb') as f1:
                soft_target = Header.pickle.load(f1)
        while 1:
            x, y, z= [],[],[]
            i = 0
            for index,name in enumerate(Header.os.listdir(feature_dir)):
                im = None
                im = Image.open(feature_dir+'/'+name)
                im = im.convert('RGB')
                im = im.resize((224,224))
                pix = list(im.getdata())
                pix = Header.np.asarray(pix)
                pix = Header.np.reshape(pix,(224,224,3))
                if(im != None):
                    im.close()
                label = int(Header.np.chararray.split(Header.np.chararray.split(name,['_'])[0][-1],['.'])[0][0])

                if gen is not None:
                    pix_tmp = Header.np.reshape(pix,(-1,224,224,3))
                    label_tmp = Header.np.reshape(label,(1,-1))
                    pix_tmp = pix_tmp.astype('float32')
                    label_tmp = label_tmp.astype('float32')
                    gen.fit(pix_tmp)
                    for x_batch,y_batch in gen.flow(pix_tmp,label_tmp,batch_size = data_gen_num):
                        for j in range(0,data_gen_num):
                            x.append(Header.np.asarray(x_batch.reshape(224,224,3)))
                            y.append(Header.np.asarray(y_batch.reshape(-1)))
                            z.append(Header.np.asarray(y_batch.reshape(-1)))
                        break
                    i += data_gen_num

                y.append(label)
                x.append(pix)
                if soft_file is not None:
                    z.append(soft_target[index])
                else:
                    z.append(label)
                i += 1
                if i == batch_size:
                    y = Header.np_utils.to_categorical(y,self.num_class)
                    z = Header.np_utils.to_categorical(z,self.num_class)
                    yield ([Header.np.array(x),Header.np.array(z)],Header.np.array(y))
                    i = 0
                    x, y, z = [], [], []
                elif(index == length - 1):
                    y = Header.np_utils.to_categorical(y,self.num_class)
                    z = Header.np_utils.to_categorical(z,self.num_class)
                    yield ([Header.np.array(x),Header.np.array(z)],Header.np.array(y))
                    i = 0
                    x, y, z = [], [], []
