import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

top1_acc = np.load("top1_acc.npy")
top5_acc = np.load("top5_acc.npy")

print(top5_acc)
print(np.mean(top5_acc))

classes = ("Apple Pie", "Baby back ribs","Baklava","Beef carpaccio","Beef tartare","Beet salad","Beignets","Bibimbap","Bread pudding","Breakfast burrito","Bruschetta","Caesar salad","Cannoli","Caprese salad","Carrot cake","Ceviche","Cheesecake","Cheese plate","Chicken curry","Chicken quesadilla","Chicken wings","Chocolate cake","Chocolate mousse","Churros","Clam chowder","Club sandwich","Crab cakes","Creme brulee","Croque madame","Cup cakes","Deviled eggs","Donuts","Dumplings","Edamame","Eggs benedict","Escargots","Falafel","Filet mignon","Fish and chips","Foie gras","French fries","French onion soup","French toast","Fried calamari","Fried rice","Frozen yogurt","Garlic bread","Gnocchi","Greek salad","Grilled cheese sandwich","Grilled salmon","Guacamole","Gyoza","Hamburger","Hot and sour soup","Hot dog","Huevos rancheros","Hummus","Ice cream","Lasagna","Lobster bisque","Lobster roll sandwich","Macaroni and cheese","Macarons","Miso soup","Mussels","Nachos","Omelette","Onion rings","Oysters","Pad thai","Paella","Pancakes","Panna cotta","Peking duck","Pho","Pizza","Pork chop","Poutine","Prime rib","Pulled pork sandwich","Ramen","Ravioli","Red velvet cake","Risotto","Samosa","Sashimi","Scallops","Seaweed salad","Shrimp and grits","Spaghetti bolognese","Spaghetti carbonara","Spring rolls","Steak","Strawberry shortcake","Sushi","Tacos","Takoyaki","Tiramisu","Tuna tartare","Waffles")

classes = np.array(classes)

best = np.argsort(top1_acc, axis=None)[-5:]
worst = np.argsort(top1_acc, axis=None)[:5]
avg = np.argsort(top1_acc, axis=None)[48:54]
idx = np.concatenate([worst, avg, best])

legend = classes[idx]
width = 0.3

sns.set()
data1 = top1_acc[idx].flatten()
data2 = top5_acc[idx].flatten()

fig, ax = plt.subplots()

ind = np.arange(len(data1))
r1 = ax.bar(ind, data1, width)
r2 = ax.bar(ind + width, data2, width)

ax.set_ylabel('accuracy')
ax.set_title('Best and worst class accuracies at epoch 50')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(classes[idx], rotation=45)
ax.legend((r1[0], r2[0]), ('Top1', 'Top5'), frameon=True, loc="lower left")
plt.show()
