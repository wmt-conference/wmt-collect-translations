import os
import glob
import ipdb
import pandas as pd

# per Million tokens in USD
pricing = {
    'Aya23': (0.8, 0.8), # https://www.together.ai/pricing
    'Claude-3.5': (3, 15), # https://www.anthropic.com/pricing
    'CommandR-plus': (3, 15), # https://cohere.com/pricing
    'Gemini-1.5-Pro': (7, 21), # https://ai.google.dev/pricing
    'GPT-4': (30, 60), # https://openai.com/api/pricing/
    'Llama3-70B': (0.9, 0.9), # https://www.together.ai/pricing
    'Mistral-Large': (4, 12), # https://mistral.ai/technology/
}

data = {}
# list all folders in wmt_translations
for system in glob.glob('wmt_translations/*'):
    sysname = system.split("/")[-1]
    data[sysname] = (0, 0)
    for filename in glob.glob(system + '/*'):
        lp = filename.split("/")[-1].split(".")[2]
        if not filename.endswith("tokens"):
            continue
        if not "no-testsuites" in filename:
            continue

        input = 0
        output = 0
        with open(filename, 'r') as f:
            for line in f:
                if "Input" in line:
                    input = line.split(": ")[1].strip()
                if "Output" in line:
                    output = line.split(": ")[1].strip()

        data[sysname] = (data[sysname][0] + int(input), data[sysname][1] + int(output))

    # if input and output is 0, remove system
    if data[sysname] == (0, 0):
        del data[sysname]

df = pd.DataFrame(data).transpose()
# rename columns
df.columns = ['Input tokens', 'Output tokens']

# divide all values by 1 million
df = df / 1000000

df['Cost'] = df.apply(lambda x: f'{x[0] * pricing[x.name][0] + x[1] * pricing[x.name][1]:.1f} $', axis=1)
# round to single decimal and append " M" to columns Input tokens and Output tokens
df['Input tokens'] = df['Input tokens'].apply(lambda x: f"{round(x, 1)} M")
df['Output tokens'] = df['Output tokens'].apply(lambda x: f"{round(x, 1)} M")

# print latex table
print(df.to_latex())