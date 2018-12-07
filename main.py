import Header
import argparse
from student import Student_model as Student
from teacher import Teacher_model as Teacher


# Execution part
def main():
     # Get image data prepared
    train_dir = './Train'
    valid_dir = './Valid'
    test_dir = './Test'

    def count_image(dir):
        count = len([1 for x in list(Header.os.scandir(dir)) if x.is_file()])
        return count

    train_num = count_image(train_dir)
    test_num = count_image(test_dir)
    valid_num = count_image(valid_dir)
    label_num = 50

    print("Data Information : ")
    print("Lable Num : ",label_num)
    print("Train Num : ",train_num)
    print("Test Num : ",test_num)
    print("Valid Num : ",valid_num)
    print("Data Num : ",train_num + test_num + valid_num)

    print("\n[Runnning Options]\n")
    print("1. Train Teacher")
    print("2. Prepare Soft Target")
    print("3. Show training log of Teacher model")
    print("4. Train Student")
    print("5. Train Student with Soft Target")
    print("6. Get the best student and test accuracy")
    print("7. Show all the training log of Student model")
    print("8. Simulation of Student Model")
    print("9. Show the temporary student model spec")
    print("10. Analyze the Student model data")
    print("\nNotice : <You should first train Teacher to get 4th option>\n")
    uinput = int(input("Press Number: "))

    # Set argument for setup
    parser_setup = Header.Set_Model_Info()
    
    args_setup = parser_setup.parse_args()

    # Option Check
    if(uinput == 1 or uinput == 2 or uinput == 3): # Teacher model
        args_setup.gpu = 1
        args_setup.model_usage = "teacher"
        args_setup.model_lang = "keras"
    elif(uinput == 4 or uinput == 5 or uinput == 6 or uinput == 7 or uinput == 8 or uinput == 9 or uinput == 10): # Student model
        args_setup.gpu = 0
        args_setup.model_usage = "student"
        args_setup.model_lang = "keras"
    else:
        print("Wrong Options. Please Start Again!")
        exit(1)

    # Setup
    Header.set_up(args_setup)

    # Set argument for training
    parser_train = Header.Set_Parameter(args_setup)
    args_train = parser_train.parse_args()

     # Teacher model
    if(args_setup.model_usage == "teacher"):
        # Train Teacher
        if(uinput == 1):
            Header.check_and_makedir(args_train.dir)
            Header.check_and_makedir(args_train.checkpoint,True)
            Header.check_and_makedir(args_train.model,True)
            Header.check_and_makedir(args_train.final,True)
            Header.check_and_makedir(args_train.log,True)

            args_train.max_overfit = 10
            teacher_model = Teacher(args_train,args_setup.model_usage,label_num)
            teacher_model.Train(args_train,train_dir,test_dir,train_num,test_num)

        # Prepare Soft Target
        elif(uinput == 2):
            Header.check_and_makedir(Header.temp_dir)
            choice = -1
            # Get Temperature
            while(choice <= 0):
                choice = float(input("Temperature >> "))
            args_train.temperature = choice

            teacher_model = Teacher(args_train,args_setup.model_usage,label_num)
            teacher_model.Save_Text(args_train.temperature,Header.temp_file) # Save the temperature file
            result = teacher_model.Evaluate(valid_dir,valid_num)
            # Save accuracy
            [train_acc,valid_acc,dummy] = teacher_model.Get_Acc()
            teacher_model.Save_Acc([train_acc,valid_acc,result[1]])
            print("Test loss : %f       Test accuracy : %f" %(result[0],result[1]))
            teacher_model.Save_soft_value(train_dir,train_num,args_train.temperature)
        else:
            teacher_model = Teacher(args_train,args_setup.model_usage,label_num)
            teacher_model.View_Loss()
            if(teacher_model.View_Train_log() == False):
                return -1
    else:
        if(uinput == 4):
            Header.check_and_makedir(args_train.dir)
            Header.check_and_makedir(args_train.checkpoint,True)
            Header.check_and_makedir(args_train.model,True)
            Header.check_and_makedir(args_train.final,True)
            Header.check_and_makedir(args_train.log,True)

            student_model = Student(args_train,args_setup.model_usage,label_num)
            student_model.Train(args_train,train_dir,valid_dir,train_num,valid_num) ##add directory
        elif(uinput == 5):
            Header.check_and_makedir(args_train.dir)
            Header.check_and_makedir(args_train.checkpoint,True)
            Header.check_and_makedir(args_train.model,True)
            Header.check_and_makedir(args_train.final,True)
            Header.check_and_makedir(args_train.log,True)

            student_model = Student(args_train,args_setup.model_usage,label_num)
            # Get the soft target's temperature data
            args_train.temperature = student_model.Load_Text(Header.temp_file)
            if(args_train.temperature == []):
                print("First train the Teacher model to get soft targets")
                exit(1)
            else:
                student_model.Define_Arg(args_train)
            # Prepare soft y data
            if(Header.os.path.isfile(args_train.softy_file) == False):
                print("First train the Teacher model to get soft targets")
                exit(1)         
            args_train.max_overfit = 15
            student_model.Train(args_train,train_dir,test_dir,train_num,test_num,True) ##add directory

        elif(uinput == 6):
            student_model = Student(args_train,args_setup.model_usage,label_num)
            teacher_model = Teacher(args_train,args_setup.model_usage,label_num)
            if(Header.os.path.isfile(args_train.softy_file) == False):
                result  = student_model.Evaluate(test_dir,args_train.softy_file,test_num) ##add directory
                print("Test accuracy : %f" %(result[1]))
            if(Header.os.path.isfile(args_train.softy_file) == True):
                [train_t_acc,valid_t_acc,test_t_acc] = student_model.Load_Text(args_train.teacher_acc_file)
                result  = student_model.Evaluate(test_dir,args_train.softy_file,test_num) ##add directory
                print("Student model Test accuracy : %f" %(result[1]))
                print("Teacher model Test accuracy : %f" %test_t_acc)
                print("# Student_params : %d" % student_model.Num_Parameter())
                print("# Teacher_params : %d" % teacher_model.Num_Parameter())
                print("Complexity_gain(Student params / Teacher params) : %f" % (float(student_model.Num_Parameter())) / float(teacher_model.Num_Parameter()))
                print("Accuracy_gain(Teacher acc - Student acc) : %f" % float(test_t_acc - result[1]))
                print("Overall_gain(1 / (Complexity_gain * Accuracy_gain)) : %f" % 1 / Header.np.mutliply(float(teacher_model.Num_Parameter()) / float(student_model.Num_Parameter()),test_t_acc - result[1]))
        elif(uinput == 7):
            student_model = Student(args_train,args_setup.model_usage,label_num)
            if(student_model.View_Train_log() == False):
                return -1
        elif(uinput == 8):
            Header.check_and_makedir(args_train.tmp,True)

            student_model = Student(args_train,args_setup.model_usage,label_num)
            teacher_model = Teacher(args_train,args_setup.model_usage,label_num)

            choice = 0
            layer = 0
            hidden_node = []
            learn_rate = 0.001
            batch = 128
            
            # Get Layer number
            while(choice <= 0):
                choice = int(input("Number of Hidden Layer >> "))
            layer = choice
            choice = -1

            # Get Node number
            for i in range(layer):
                while(choice <= 0):
                    choice = int(input("Number of Hidden node of layer %s >> " % str(i+1)))
                hidden_node.append(choice)
                choice = -1

            # Get Learning rate
            while(choice <= 0):
                choice = float(input("Learning_rate >> "))
            learn_rate = choice
            choice = -1

            # Get Batch size
            while(choice <= 0):
                choice = int(input("Batch size >> "))
            batch = choice
            choice = " "

            soft_target = []
            if(Header.os.path.isfile(args_train.softy_file) == True):
                # If the model has to use the soft target
                while(choice.lower() != "yes" and choice.lower() != "no"):
                    choice = input("Soft Target Found. Shall we use it? (Yes or No) >> ")
                if(choice.lower() == "yes"):
                    args_train.temperature = student_model.Load_Text(Header.temp_file)
                    if(args_train.temperature == []):
                        print("Temperature information is lost! Please try option 2 again")
                        exit(1)
                    else:
                        student_model.Define_Arg(args_train)
                    soft_target = soft_train

            [acc,num_param] = student_model.Fit_SingleModel(train_dir,valid_dir,test_dir,args_train.softy_file,train_num,valid_num,test_num,hidden_node,learn_rate,batch) ##add directory
            [train_t_acc,valid_t_acc,test_t_acc] = student_model.Load_Text(args_train.teacher_acc_file)

            complexity_gain = Header.np.divide(float(num_param) , float(teacher_model.Num_Parameter()))
            accuracy_gain = float(test_t_acc - acc)
            overall_gain = 1 / Header.np.multiply(complexity_gain,accuracy_gain)
            student_model.Temp_file3_add([complexity_gain,accuracy_gain,overall_gain])

            print("Student model Test accuracy : %f" %(acc))
            print("Teacher model Test accuracy : %f" %test_t_acc)
            print("# Student_params : %d" % num_param)
            print("# Teacher_params : %d" % teacher_model.Num_Parameter())
            print("Complexity_gain(Student params / Teacher params) : %f" % complexity_gain)
            print("Accuracy_gain(Teacher acc - Student acc) : %f" % accuracy_gain)
            print("Overall_gain(1 / (Accuracy_gain * Complexity_gain)) : %f" % overall_gain)
        elif(uinput == 9):
            pass
        else:
            pass

    return 0

if __name__ == '__main__':
    main()

