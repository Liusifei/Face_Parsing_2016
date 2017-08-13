function EVAL = evaluate(ACTUAL,PREDICTED,l)
% This fucntion evaluates the performance of a classification model by 
% calculating the common performance measures: Accuracy, Sensitivity, 
% Specificity, Precision, Recall, F-Measure, G-mean.
% Input: ACTUAL = Column matrix with actual class labels of the training
%                 examples
%        PREDICTED = Column matrix with predicted class labels by the
%                    classification model
%        l: classes e.g. 1~3
% Output: EVAL = Row matrix with all the performance measures

N = length(ACTUAL(:));
accuracy = length(find(ACTUAL(:)==PREDICTED(:)))/N;
idx = (ACTUAL()==l);
nidx = (PREDICTED()==l);

ACTUAL(idx)=1;ACTUAL(~idx)=0;
PREDICTED(nidx)=1;PREDICTED(~nidx)=0;

p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));

tp = sum(ACTUAL(idx)==PREDICTED(idx));
tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp = n-tn;
fn = p-tp;

tp_rate = tp/p;
tn_rate = tn/n;

sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);

EVAL = [accuracy sensitivity specificity precision recall f_measure gmean];

