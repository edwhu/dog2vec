import pandas as pd
import matplotlib.pyplot as plt
# look at owners of most popular dog
def main():
    plt.style.use('ggplot')
    # Import the CSV into a dataframe
    df = pd.read_csv('data.csv');
    # Drop null rows
    df = df[pd.notnull(df['ALTER'])]
    # df = create_minage_maxage_cols(df);
    # print(top_ten_dogs(df))
    top_three = top_three_dogs(df)
    # print(top_dog_name, top_dog_freq)
    # print(top_three.index)
    df = create_minage_maxage_cols(df)
    top_dog_owners_age_histogram(df, list(top_three.index.values))
# calculate top dog
def top_three_dogs(df):
    dog_types = df['RASSE1']
    top_dog = dog_types.value_counts()
    return top_dog[0:3]

def create_minage_maxage_cols(df):
    print('Creating Minage Maxage columns')
    # create age_min and age_max columns
    age_series = df['ALTER']
    age_min_list = [];
    age_max_list = [];
    for index, value in age_series.iteritems():
        minage,maxage = value.split("-")
        age_min_list.append(int(minage))
        age_max_list.append(int(maxage))
    # add age_min and age_max columns
    return df.assign(age_min=age_min_list, age_max=age_max_list)
# show some trends of the top dog
def top_dog_owners_age_histogram(df, top_dog_names):
    # print(top_dog_names)
    # all rows where top dog
    top_dog_rows = df.loc[df['RASSE1'] == top_dog_names[0]]
    sec_dog_rows = df.loc[df['RASSE1'] == top_dog_names[1]]
    tri_dog_rows = df.loc[df['RASSE1'] == top_dog_names[2]]
    # histogram of owner Birth dates
    top_dog_bdays = top_dog_rows['age_min']
    sec_dog_bdays = sec_dog_rows['age_min']
    tri_dog_bdays = tri_dog_rows['age_min']

    plt.hist(top_dog_bdays, bins=range(0, 100 + 10, 10), histtype='stepfilled', label=top_dog_names[0],alpha=1.0, normed=True)

    plt.hist(sec_dog_bdays, bins=range(0, 100 + 10, 10), histtype='stepfilled', label=top_dog_names[1],alpha=0.66, normed=True)

    # plt.hist(tri_dog_bdays, bins=range(0, 100 + 10, 10), histtype='step', label=top_dog_names[2],alpha=1.0, normed=True)
    plt.title("Age of Dog Owners in Zurich")
    plt.xlabel("Year")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.show()
if __name__ == "__main__":
    main()
