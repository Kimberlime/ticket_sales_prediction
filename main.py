import os
import pandas as pd


def preprocess_data(data_path, save_file):
    """ Pre-process data and returns it.
    :param data_path: The path to find data in
    :param save_file: The boolean which checks whether to save data as a file(preprocessed_data.csv)
    :return: The pandas DataFrame with pre-processed data
    """
    concert_list = pd.read_csv(os.path.join(data_path, 'concert_list.csv'), parse_dates=['closing_date'])
    artist_list = pd.read_csv(os.path.join(data_path, 'artist_list.csv'))
    vlive_data = pd.read_csv(os.path.join(data_path, 'vlive_data.csv'), parse_dates=['upload_date'])
    mv_data = pd.read_csv(os.path.join(data_path, 'mv_data.csv'), parse_dates=['upload_date'])
    twitter_data = pd.read_csv(os.path.join(data_path, 'twitter_data.csv'), parse_dates=['upload_date'])

    processed_data = pd.merge(concert_list, artist_list, on='artist')

    vlive_columns = ['follower', 'playtime', 'view_count', 'like_count', 'comment_count']
    vlive_columns_modified = ['v_' + c for c in vlive_columns]

    mv_columns = ['view_count', 'like_count', 'dislike_count', 'comment_count']
    mv_columns_modified = ['m_' + c for c in mv_columns]

    twitter_columns = ['follower', 'total_tweet', 'like_count', 'retweet_count', 'comment_count']
    twitter_columns_modified = ['t_' + c for c in twitter_columns]

    total_columns = vlive_columns_modified + mv_columns_modified + twitter_columns_modified
    processed_data = pd.concat([processed_data, pd.DataFrame(columns=total_columns)], sort=False)
    for index, row in processed_data.iterrows():
        artist_vlive_data = vlive_data.loc[vlive_data['artist'] == row['artist']]
        valid_vlive_data = artist_vlive_data[artist_vlive_data['upload_date'] > row['closing_date']]
        valid_vlive_data = valid_vlive_data.sort_values(by='upload_date', ascending=False)

        processed_data.at[index, vlive_columns_modified[0]] = valid_vlive_data.iloc[0][vlive_columns[0]]
        processed_data.at[index, vlive_columns_modified[1]] = valid_vlive_data[vlive_columns[1]].sum()
        processed_data.at[index, vlive_columns_modified[2]] = valid_vlive_data[vlive_columns[2]].sum()
        processed_data.at[index, vlive_columns_modified[3]] = valid_vlive_data[vlive_columns[3]].sum()
        processed_data.at[index, vlive_columns_modified[4]] = valid_vlive_data[vlive_columns[4]].sum()

        artist_mv_data = mv_data.loc[mv_data['artist'] == row['artist']]
        valid_mv_data = artist_mv_data[artist_mv_data['upload_date'] > row['closing_date']]
        valid_mv_data = valid_mv_data.sort_values(by='upload_date', ascending=False)

        processed_data.at[index, mv_columns_modified[0]] = valid_mv_data[mv_columns[0]].sum()
        processed_data.at[index, mv_columns_modified[1]] = valid_mv_data[mv_columns[1]].sum()
        processed_data.at[index, mv_columns_modified[2]] = valid_mv_data[mv_columns[2]].sum()
        processed_data.at[index, mv_columns_modified[3]] = valid_mv_data[mv_columns[3]].sum()

        artist_twitter_data = twitter_data.loc[twitter_data['artist'] == row['artist']]
        valid_twitter_data = artist_twitter_data[artist_twitter_data['upload_date'] > row['closing_date']]
        valid_twitter_data = valid_twitter_data.sort_values(by='upload_date', ascending=False)

        processed_data.at[index, twitter_columns_modified[0]] = valid_twitter_data.iloc[0][twitter_columns[0]]
        processed_data.at[index, twitter_columns_modified[1]] = valid_twitter_data.iloc[1][twitter_columns[0]]
        processed_data.at[index, twitter_columns_modified[2]] = valid_twitter_data[twitter_columns[2]].sum()
        processed_data.at[index, twitter_columns_modified[3]] = valid_twitter_data[twitter_columns[3]].sum()
        processed_data.at[index, twitter_columns_modified[4]] = valid_twitter_data[twitter_columns[4]].sum()

    if save_file:
        processed_data.to_csv('preprocessed_data.csv')

    return processed_data


def main():
    processed_data = preprocess_data('data', True)


if __name__ == "__main__":
    main()
