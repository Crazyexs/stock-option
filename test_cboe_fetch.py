import en_option_v3 as opt
df = opt.fetch_cboe_chain("SOFI")
print("CBOE fetch returned shape:", df.shape)
