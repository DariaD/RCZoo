# import the required library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# load the dataset
df = pd.read_csv("/home/pinecone/Dropbox/MyPaper/RC_Zoo/table2_almost_full.csv", sep="&", header=0)


def plot_len(plot_name, data):
      fig1, ax1 = plt.subplots()
      ax1.set_title(plot_name,)
      to_plot = []
      for x in data:
          if "-" == x:
             continue
          to_plot.append(float(x))
      print(to_plot)
      ax1.boxplot(to_plot)
      plt.savefig('pic/{}.png'.format("_".join(plot_name.split())))
      plt.show()


def plot_vocabulary(plot_name, data):
      fig1, ax1 = plt.subplots()
      ax1.set_title(plot_name)
      print(data)
      to_plot = []
      for x in data:
          if len(x.strip()) > 0:
                to_plot.append(int(x))
      print(to_plot)
      ax1.boxplot(to_plot)
      plt.savefig('pic/{}.png'.format("_".join(plot_name.split())))
      plt.show()


plot_len("Average Question Length", df["question_len"])
plot_len("Average Passage Length", df["passage_len"])
plot_len("Average Answer Length", df["answer_len"])
plot_vocabulary("Vocabulary", df["vocab_size"])


# #print(df.head())
# to_plot = []
# for x in df["question_len"]:
#     #x = x.strip().replace(",", "")
#     #if len(x) > 0:
#         print(x)
#         to_plot.append(float(x))
# #data = [int(x.strip().replace(",", "")) for x in df["vocab_size"]]
# # display 5 rows of dataset
# #print(data)
#
# fig1, ax1 = plt.subplots()
# ax1.set_title('Basic Plot')
# ax1.boxplot(to_plot)
# plt.show()
# print()



