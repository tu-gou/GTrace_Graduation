import pandas as pd
import matplotlib.pyplot as plt

loss_df = pd.read_csv('../models/gtrace/losses.csv')
plt.plot(loss_df['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
