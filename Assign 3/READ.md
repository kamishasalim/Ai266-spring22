<h1 align="center">Assignment - 3</h1>

## Group Members
| Student ID | Name | ML Technique |
| :---: | :---:  | :---:  |
| 11041 | Bilal Shoukat | Naive Bayes - Lidstone Smoothing |
| 11070 | Kamisha Salim | KNN |
| 62803 | Murtaza Memon | Naive Bayes - Laplace Smoothing |
| 64055 | Muhammad Kashan | Perceptron |

## Problems Faced in the Assignment
### Bilal Shoukat (Lidstone Smoothing)
Lidstone smoothing, is a technique used to smooth categorical data. Given a set of observation counts from a -dimensional multinomial distribution with. It is very difficult to increase the accuracy of this model even after increasing the number of states in the code. After some tweaking with the values and even deleting column f_27 which had irrelevent values, the score increased by just 0.00003%. Training this model was not slow at all but the difficult part was increasing its accuracy.

### Kamisha Salim (KNN)
The k-nearest neighbors (KNN) algorithm is a simple, supervised machine learning algorithm that can be used to solve both classification and regression problems. It's easy to implement and understand. </br>
The accuracy of this model was 50%-53% percent when the number of neighbours were 15 but after increasing just one more neighbour i.e. 16, the accuracy of this model significantly changed to 65%. Which means that increasing the amount of neighbours will also increase the accuracy of the model. The problem of accuracy was also solved by selecting the columns with the best variance (f_22, f_25, f_19, f_24, f_21, f_26) and also deleting column f_27 because it was irrelevent. The major drawback of KNN is becoming drastically slow as the size of the data in use grows. The main problem was that it takes a lot of time to train this model.

## Screenshot of Kaggle score
### Bilal Shoukat (Lidstone Smoothing)
![Screenshot 2022-05-21 102450](https://user-images.githubusercontent.com/63594764/169636928-78981648-d542-456d-b56a-83dc17e35bbb.png)

### Kamisha Salim (KNN)
![Screenshot_2](https://user-images.githubusercontent.com/99355356/169399555-9586bd8b-8ab4-430e-8e04-31f3f1ed060c.png)

### Murtaza Memon (Laplace Smoothing)
![image](https://user-images.githubusercontent.com/41837489/169532492-e192faa4-e219-4930-9b46-ec0cd09cf733.png)


### Muhammad Kashan (Perceptron)
