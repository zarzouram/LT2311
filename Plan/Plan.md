# End-to-end Argument Mining (AM)

- [1. Goal](#1-goal)
- [2. Problem description](#2-problem-description)
- [3. Schedule](#3-schedule)
- [4. Supervisors](#4-supervisors)
- [References](#references)

## 1. Goal

Automatic identification and extraction of arguments presented in natural language texts.

## 2. Problem description

AM deals with finding argumentation structure in unstructured text. This proceess involves mainly two stages[[1]](#1)[[2]](#2):

1. Arguments extraction, which involves
   1. Arguments Identification: Component segmentation of text to identify arguments locations and thier boundery within the text.
   2. Arguments components Indentification: Classify identified arguemnts into different classes such as claim,  premise

2. Relations prediction, which involves finding the relation between argument components such as against, support, etc.

Example: [[1]](#1).

Since it killed many marine lives *(Premise)*,tourism has threatened nature *(Claim)*.

## 3. Schedule

The project is segemented into five major steps:

1. Survey
2. Impelementation
3. Result analysis
4. Re-Implementation and re-analysis (if needed)
5. Reporting and Presentation

Refer to the appendix A for the time schedule.

## 4. Supervisors

1. Asad Sayeed, asad.sayeed@gu.se
2. Axel Almquist, axel.almquist@gu.se


## References

<a id="1">[1]</a>
Eger, S., Daxenberger, J., & Gurevych, I. (2017). Neural end-to-end learning for computational argumentation mining. arXiv preprint arXiv:1704.06104.

<a id="2">[2]</a>
Cabrio, E., & Villata, S. (2018, July). Five Years of Argument Mining: a Data-driven Analysis. In IJCAI (Vol. 18, pp. 5427-5433).