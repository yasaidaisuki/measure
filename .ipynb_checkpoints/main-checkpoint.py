import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.ticker as ticker

fig, ax = plt.subplots()      

fig.set_size_inches(12, 5)


header =['frequency','raw']

#figure 
ax.set_title("Frequency Response")
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('dBr')

# axis scale & ticks
ax.yaxis.set_major_locator(ticker.MaxNLocator())
plt.xscale('log')
plt.xticks([0.1, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], [0, 20, 50, 100, 200, 500, '1k', '2k', '5k', '10k', '20k'])
plt.xlim(20,20000)
df = pd.read_csv('./measurements/oratory1990/over-ear/Focal Elex.csv', usecols=header)  

plt.plot(df.frequency,df.raw)
ax.grid(True)

plt.show()
