import psycopg2
import numpy as np

#conn = psycopg2.connect("dbname=postgres user=postgres password=123456")


def create_playlist(listened_song_ids, songs_per_song=2):
    """Create a playlist with recommended songs for each listened song."""
    playlist = []

    # Connect to your PostgreSQL database
    conn = psycopg2.connect("dbname=postgres user=postgres password=123456")
    cur = conn.cursor()

    # Find recommended songs for each listened song
    for song_id in listened_song_ids:
        cur.execute("SELECT features FROM music WHERE id = %s", (song_id,))
        result = cur.fetchone()
        if result:
            # SQL query to find similar songs with Euclidean distance
            query = """
            SELECT id, title
            FROM music
            WHERE id != %s
            ORDER BY features <-> CAST(%s AS vector)
            LIMIT %s;
            """
            #<-> = Euclidean distance
                  
            '''            
            # SQL query to find similar songs using angular distance
            query = """
            SELECT id, title
            FROM music
            WHERE id != %s
            ORDER BY features <#> CAST(%s AS vector)
            LIMIT %s;
            """'''

            # Execute the query and fetch the results
            cur.execute(query, (song_id, result[0], songs_per_song))
            recommended_songs = cur.fetchall()

            # Add the recommended songs to the playlist
            playlist.extend(recommended_songs)

    # Close the cursor and connection
    cur.close()
    conn.close()

    return playlist

# Example usage
listened_song_ids = [4]
playlist = create_playlist(listened_song_ids, songs_per_song=2)
print("Recommended Playlist:")
for song in playlist:
    print(song)
