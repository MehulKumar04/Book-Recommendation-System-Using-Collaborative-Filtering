import streamlit as st
import pickle
import numpy as np

# --- 1. Load Pre-computed Data ---
# Load the dataframes and similarity score matrix from the pickle files.
# Using st.cache_data ensures these heavy objects are loaded only once.
try:
    @st.cache_data
    def load_data():
        # Load all components needed for the app
        popular_df = pickle.load(open('popular.pkl', 'rb'))
        pt = pickle.load(open('pt.pkl', 'rb'))
        books = pickle.load(open('books.pkl', 'rb'))
        similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))
        return popular_df, pt, books, similarity_scores

    popular_df, pt, books, similarity_scores = load_data()

except FileNotFoundError:
    st.error("One or more required data files (popular.pkl, pt.pkl, books.pkl, similarity_scores.pkl) not found.")
    st.error("Please ensure you have run the data dumping script successfully and the files are in the same directory.")
    st.stop()


# --- 2. Recommendation Function ---
def recommend(book_name):
    """
    Finds the top 5 most similar books based on the cosine similarity score.
    
    :param book_name: The title of the book to find recommendations for.
    :return: A list of recommended books, each containing [Title, Author, Image URL].
    """
    
    # Check if the book exists in the pivot table index
    if book_name not in pt.index:
        st.warning(f"Book '{book_name}' not found in the trained database for recommendation.")
        return []

    # Get the index of the book
    index = np.where(pt.index == book_name)[0][0]
    
    # Get the top 5 similar items (excluding itself)
    # The [1:6] slicing gets 5 items
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]
    
    recommended_data = []
    for i in similar_items:
        # Fetch book details using the index (i[0]) from the pivot table
        book_title = pt.index[i[0]]
        
        # Filter the original books DataFrame to get author and image URL
        temp_df = books[books['Book-Title'] == book_title].drop_duplicates('Book-Title')
        
        if not temp_df.empty:
            item = {
                'title': temp_df['Book-Title'].values[0],
                'author': temp_df['Book-Author'].values[0],
                'image_url': temp_df['Image-URL-M'].values[0],
                'similarity_score': i[1]
            }
            recommended_data.append(item)
    
    return recommended_data


# --- 3. Streamlit App Layout ---

st.set_page_config(layout="wide")

st.title(':books: Book Recommendation System')

# --- Tab View for Organization ---
tab1, tab2 = st.tabs(["Top Popular Books", "Get Recommendations"])


with tab1:
    st.header("Top 50 Most Popular Books")
    st.markdown("These books have received at least 250 ratings and are ranked by their average rating.")
    
    # Display the popular books in columns
    cols = st.columns(5)
    
    for i in range(50):
        # Determine the column for the current card (0 to 4)
        col_index = i % 5
        
        with cols[col_index]:
            book = popular_df.iloc[i]
            
            # Create a nice container for each book
            with st.container(border=True):
                st.subheader(f"{i+1}. {book['Book-Title']}")
                st.write(f"**Author:** {book['Book-Author']}")
                st.write(f"**Avg Rating:** {book['avg_rating']:.2f} / 10")
                st.write(f"**Total Ratings:** {book['num_rating']}")
                
                # Display the image with a fallback in case the URL is bad
                st.image(book['Image-URL-M'], width=150, 
                         caption=book['Book-Title'])

with tab2:
    st.header("Collaborative Filtering Recommendations")
    st.markdown("Select a book you like, and we'll suggest 5 similar books based on user rating patterns.")
    
    # Dropdown selector for the user
    # Convert index to a list for selection
    book_titles_list = pt.index.tolist()
    
    selected_book_name = st.selectbox(
        'Start typing a book title here:',
        options=book_titles_list,
        index=None,
        placeholder="Select a book...",
    )
    
    st.markdown("---")

    if st.button('Show Recommendations', use_container_width=True):
        if selected_book_name:
            with st.spinner('Fetching recommendations...'):
                recommended_books = recommend(selected_book_name)

            if recommended_books:
                st.success(f"Showing recommendations for: **{selected_book_name}**")
                
                # Display recommendations in 5 columns
                rec_cols = st.columns(5)
                
                for j, book in enumerate(recommended_books):
                    with rec_cols[j]:
                        with st.container(border=True):
                            st.image(book['image_url'], width=150, caption=book['title'])
                            st.caption(f"**Similarity Score:** {book['similarity_score']:.3f}")
                            st.subheader(book['title'])
                            st.write(f"By: **{book['author']}**")

            else:
                st.warning("Could not find suitable recommendations for this book.")
        else:
            st.warning("Please select a book to get recommendations.")
