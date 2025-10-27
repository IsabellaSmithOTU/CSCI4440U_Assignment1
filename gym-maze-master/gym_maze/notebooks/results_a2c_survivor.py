import pandas as pd 
import matplotlib.pyplot as plt  #not working

#loading csv 
df = pd.read_csv('../src/logs/eval_a2c_survivor_survivor.csv')
df.columns = df.columns.str.strip() #space issue if any 

df.apply(pd.to_numeric) #errors about values so i added this
#learning curve /reward

df.plot(x='episode', y='reward',  figsize=(7,4), grid=True, title='Learning curve')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.tight_layout()
plt.savefig('learningcurve_a2c_survivor.png')
plt.show()

#step per ep
df.plot(x='episode', y=['success', 'truncated'],  figsize=(7,4), grid=True, title='Success/Truncated per episode')
plt.xlabel('Episode')
plt.ylabel('Rate')
plt.tight_layout()
plt.savefig('successtrunc_a2c_survivor.png')
plt.show()


#success by the truncated rate over the the episodes

#coverage per ep
df.plot(x='episode', y='coverage',  figsize=(7,4), grid=True, title='Coverage per ep')
plt.xlabel('Episode')
plt.ylabel('Rate')
plt.tight_layout()
plt.savefig('covgperep_a2c_survivorr.png')
plt.show()


'''
plt.figure(figsize=(7,4))
plt.plot(df['episode'], df['reward'], marker='o')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Learning curve')
plt.grid(True)
plt.show()
'''