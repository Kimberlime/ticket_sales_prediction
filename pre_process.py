import os
import pandas as pd


def one_hot_encode(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Applies one-hot encoding to a column of a DataFrame and returns the DataFrame.
    :param data: The pandas DataFrame which has the target column
    :param column_name: The column to apply one-hot encoding to
    :return: The pandas DataFrame with one-hot encoded data
    """
    unique_columns = sorted(data[column_name].unique())
    columns_with_prefix = [column_name + '_' + str(c) for c in unique_columns]
    encoded_data = pd.concat([data, pd.DataFrame(columns=columns_with_prefix)], sort=False)

    continent = encoded_data.pop(column_name)
    for i in range(len(unique_columns)):
        encoded_data[columns_with_prefix[i]] = (continent == unique_columns[i]) * 1.0

    return encoded_data


def pre_process_data(data_path: str, save_file: bool) -> pd.DataFrame:
    """
    Pre-processes data and returns it.
    :param data_path: The path to find data in
    :param save_file: The boolean which checks whether to save data as a file(preprocessed_data.csv)
    :return: The pandas DataFrame with pre-processed data
    """
    concert_list = pd.read_csv(os.path.join(data_path, 'concert_list.csv'), parse_dates=['closing_date'])
    artist_list = pd.read_csv(os.path.join(data_path, 'artist_list.csv'))
    vlive_data = pd.read_csv(os.path.join(data_path, 'vlive_data.csv'), parse_dates=['upload_date'])
    mv_data = pd.read_csv(os.path.join(data_path, 'mv_data.csv'), parse_dates=['upload_date'])
    twitter_data = pd.read_csv(os.path.join(data_path, 'twitter_data.csv'), parse_dates=['upload_date'])

    processed_data = one_hot_encode(concert_list, 'continent')
    processed_data = one_hot_encode(processed_data, 'city')

    processed_data = pd.merge(processed_data, artist_list, on='artist')
    processed_data = one_hot_encode(processed_data, 'gender')

    vlive_columns = ['follower', 'playtime', 'view_count', 'like_count', 'comment_count']
    vlive_columns_modified = ['v_' + c for c in vlive_columns]

    mv_columns = ['view_count', 'like_count', 'dislike_count', 'comment_count']
    mv_columns_modified = ['m_' + c for c in mv_columns]

    twitter_columns = ['follower', 'total_tweet', 'like_count', 'retweet_count', 'comment_count']
    twitter_columns_modified = ['t_' + c for c in twitter_columns]

    total_columns = vlive_columns_modified + mv_columns_modified + twitter_columns_modified
    processed_data = pd.concat([processed_data, pd.DataFrame(columns=total_columns)], sort=False)
    for i, row in processed_data.iterrows():
        artist_vlive_data = vlive_data.loc[vlive_data['artist'] == row['artist']]
        valid_vlive_data = artist_vlive_data[artist_vlive_data['upload_date'] > row['closing_date']]
        valid_vlive_data = valid_vlive_data.sort_values(by='upload_date', ascending=False)

        processed_data.at[i, vlive_columns_modified[0]] = valid_vlive_data.iloc[0][vlive_columns[0]]
        processed_data.at[i, vlive_columns_modified[1]] = valid_vlive_data[vlive_columns[1]].sum()
        processed_data.at[i, vlive_columns_modified[2]] = valid_vlive_data[vlive_columns[2]].sum()
        processed_data.at[i, vlive_columns_modified[3]] = valid_vlive_data[vlive_columns[3]].sum()
        processed_data.at[i, vlive_columns_modified[4]] = valid_vlive_data[vlive_columns[4]].sum()

        artist_mv_data = mv_data.loc[mv_data['artist'] == row['artist']]
        valid_mv_data = artist_mv_data[artist_mv_data['upload_date'] > row['closing_date']]
        valid_mv_data = valid_mv_data.sort_values(by='upload_date', ascending=False)

        processed_data.at[i, mv_columns_modified[0]] = valid_mv_data[mv_columns[0]].sum()
        processed_data.at[i, mv_columns_modified[1]] = valid_mv_data[mv_columns[1]].sum()
        processed_data.at[i, mv_columns_modified[2]] = valid_mv_data[mv_columns[2]].sum()
        processed_data.at[i, mv_columns_modified[3]] = valid_mv_data[mv_columns[3]].sum()

        artist_twitter_data = twitter_data.loc[twitter_data['artist'] == row['artist']]
        valid_twitter_data = artist_twitter_data[artist_twitter_data['upload_date'] > row['closing_date']]
        valid_twitter_data = valid_twitter_data.sort_values(by='upload_date', ascending=False)

        processed_data.at[i, twitter_columns_modified[0]] = valid_twitter_data.iloc[0][twitter_columns[0]]
        processed_data.at[i, twitter_columns_modified[1]] = valid_twitter_data.iloc[1][twitter_columns[0]]
        processed_data.at[i, twitter_columns_modified[2]] = valid_twitter_data[twitter_columns[2]].sum()
        processed_data.at[i, twitter_columns_modified[3]] = valid_twitter_data[twitter_columns[3]].sum()
        processed_data.at[i, twitter_columns_modified[4]] = valid_twitter_data[twitter_columns[4]].sum()

    if save_file:
        processed_data.to_csv('preprocessed_data.csv', index=False)

    return processed_data
