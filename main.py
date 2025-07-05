import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Data Sosial Media",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Memuat dan memproses data"""
    try:
        # Load data
        users = pd.read_csv('https://raw.githubusercontent.com/cayaaa14/data-real/refs/heads/main/user_table.csv')
        friends = pd.read_csv('https://raw.githubusercontent.com/cayaaa14/data-real/refs/heads/main/friends_table.csv')
        posts = pd.read_csv('https://raw.githubusercontent.com/cayaaa14/data-real/refs/heads/main/posts_table.csv')
        reactions = pd.read_csv('https://raw.githubusercontent.com/cayaaa14/data-real/refs/heads/main/reactions_table.csv')
        
        # Data cleaning
        users['Subscription Date'] = pd.to_datetime(users['Subscription Date'], unit='s')
        posts['Post Date'] = pd.to_datetime(posts['Post Date'], unit='s')
        
        # Clean reactions data
        reactions = reactions.dropna(subset=['User'])
        mode_rt = reactions['Reaction Type'].mode()[0]
        reactions['Reaction Type'] = reactions['Reaction Type'].fillna(mode_rt)
        median_rd = reactions['Reaction Date'].median()
        reactions['Reaction Date'] = reactions['Reaction Date'].fillna(median_rd)
        reactions['Reaction Date'] = pd.to_datetime(reactions['Reaction Date'], unit='s')
        
        # Remove duplicates
        users = users.drop_duplicates()
        friends = friends.drop_duplicates()
        posts = posts.drop_duplicates()
        reactions = reactions.drop_duplicates()
        
        return users, friends, posts, reactions
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

@st.cache_data
def create_integrated_dataset(users, friends, posts, reactions):
    """Membuat dataset terintegrasi"""
    # Base dari users
    users = users.copy()
    users['user_id'] = range(1, len(users) + 1)
    integrated_data = users[['user_id', 'Name', 'Surname', 'Age', 'Subscription Date']].copy()
    
    # Data teman
    all_friends = pd.concat([
        friends[['Friend 1']].rename(columns={'Friend 1': 'user_id'}),
        friends[['Friend 2']].rename(columns={'Friend 2': 'user_id'})
    ])
    friend_stats = all_friends.value_counts('user_id').reset_index()
    friend_stats.columns = ['user_id', 'friend_count']
    
    # Data posting
    posts = posts.copy()
    posts = posts.rename(columns={'User': 'user_id'})
    posts['post_id'] = range(1, len(posts) + 1)
    post_stats = posts.groupby('user_id').agg({'post_id': 'count'}).reset_index()
    post_stats.columns = ['user_id', 'post_count']
    
    # Data reaksi diberikan
    reactions = reactions.copy()
    reactions = reactions.rename(columns={'User': 'user_id'})
    reactions['reaction_id'] = range(1, len(reactions) + 1)
    reactions_given = reactions.groupby('user_id').agg({'reaction_id': 'count'}).reset_index()
    reactions_given.columns = ['user_id', 'reactions_given']
    
    # Simulasi reaksi terhadap posting
    reactions['post_id'] = reactions['reaction_id'] % len(posts) + 1
    post_reactions = posts.merge(reactions, on='post_id', how='inner')
    reactions_received = post_reactions.groupby('user_id_x').agg({'user_id_y': 'count'}).reset_index()
    reactions_received.columns = ['user_id', 'reactions_received']
    
    # Gabungkan semua data
    integrated_data = integrated_data.merge(friend_stats, on='user_id', how='left')
    integrated_data = integrated_data.merge(post_stats, on='user_id', how='left')
    integrated_data = integrated_data.merge(reactions_given, on='user_id', how='left')
    integrated_data = integrated_data.merge(reactions_received, on='user_id', how='left')
    
    # Bersihkan missing values
    integrated_data.fillna(0, inplace=True)
    
    # Tambahan kolom analisis
    integrated_data['age_group'] = pd.cut(
        integrated_data['Age'],
        bins=[0, 20, 30, 40, 50, 100],
        labels=['<20', '20-29', '30-39', '40-49', '50+']
    )
    integrated_data['registration_year'] = pd.to_datetime(integrated_data['Subscription Date']).dt.year
    integrated_data['is_active_poster'] = integrated_data['post_count'] > 0
    integrated_data['is_social'] = integrated_data['friend_count'] > 0
    integrated_data['engagement_ratio'] = integrated_data['reactions_received'] / (integrated_data['post_count'] + 1)
    integrated_data['total_activity'] = (integrated_data['friend_count'] + 
                                       integrated_data['post_count'] + 
                                       integrated_data['reactions_given'])
    
    return integrated_data, posts, reactions

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Analisis Data Sosial Media</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Menggabungkan Informasi Pengguna, Jaringan, dan Interaksi</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Memuat dan memproses data...'):
        users, friends, posts, reactions = load_data()
        if users is None:
            st.error("Gagal memuat data. Silakan periksa koneksi internet Anda.")
            return
        
        integrated_data, posts_processed, reactions_processed = create_integrated_dataset(users, friends, posts, reactions)
    
    # Sidebar untuk navigasi
    st.sidebar.title("🎛️ Navigasi")
    sections = [
        "📈 Overview & Statistik",
        "👥 Analisis Demografi",
        "🏆 Top Performers",
        "💝 Analisis Reaksi",
        "📊 Distribusi Aktivitas",
        "🔗 Analisis Korelasi",
        "🎯 Insights Mendalam"
    ]
    
    selected_section = st.sidebar.selectbox("Pilih Bagian Analisis:", sections)
    
    # Overview & Statistik
    if selected_section == "📈 Overview & Statistik":
        st.markdown('<h2 class="sub-header">📈 Overview & Statistik Umum</h2>', unsafe_allow_html=True)
        
        # Metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Total Pengguna", f"{len(users):,}", delta=None)
        with col2:
            st.metric("🤝 Total Pertemanan", f"{len(friends):,}", delta=None)
        with col3:
            st.metric("📝 Total Postingan", f"{len(posts):,}", delta=None)
        with col4:
            st.metric("💝 Total Reaksi", f"{len(reactions):,}", delta=None)
        
        # Detailed statistics
        st.subheader("📊 Statistik Detail")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**👥 Statistik Pengguna:**")
            st.write(f"• Usia rata-rata: {users['Age'].mean():.1f} tahun")
            st.write(f"• Usia median: {users['Age'].median():.1f} tahun")
            st.write(f"• Rentang usia: {users['Age'].min()}-{users['Age'].max()} tahun")
            
            st.write("**🤝 Statistik Pertemanan:**")
            all_friends = pd.concat([
                friends[['Friend 1']].rename(columns={'Friend 1': 'user_id'}),
                friends[['Friend 2']].rename(columns={'Friend 2': 'user_id'})
            ])
            avg_friends = all_friends.groupby('user_id').size().mean()
            st.write(f"• Rata-rata teman per pengguna: {avg_friends:.1f}")
            st.write(f"• Pengguna dengan teman: {all_friends['user_id'].nunique():,}")
        
        with col2:
            st.write("**📝 Statistik Postingan:**")
            active_posters = posts['User'].nunique()
            st.write(f"• Pengguna aktif posting: {active_posters:,}")
            st.write(f"• Persentase active posters: {active_posters / len(users) * 100:.1f}%")
            avg_posts = posts.groupby('User').size().mean()
            st.write(f"• Rata-rata post per active user: {avg_posts:.1f}")
            
            st.write("**💝 Statistik Reaksi:**")
            users_giving_reactions = reactions['User'].nunique()
            st.write(f"• Users memberi reaksi: {users_giving_reactions:,}")
            st.write(f"• Persentase pemberi reaksi: {users_giving_reactions / len(users) * 100:.1f}%")
    
    # Analisis Demografi
    elif selected_section == "👥 Analisis Demografi":
        st.markdown('<h2 class="sub-header">👥 Analisis Demografi</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribusi Kelompok Usia
            age_distribution = integrated_data['age_group'].value_counts().sort_index()
            fig = px.bar(
                x=age_distribution.index,
                y=age_distribution.values,
                title="Distribusi Pengguna per Kelompok Usia",
                labels={'x': 'Kelompok Usia', 'y': 'Jumlah Pengguna'},
                color=age_distribution.values,
                color_continuous_scale='viridis'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Histogram Usia
            fig = px.histogram(
                integrated_data,
                x='Age',
                title="Distribusi Usia Pengguna (Histogram)",
                labels={'Age': 'Usia (tahun)', 'count': 'Jumlah Pengguna'},
                nbins=30
            )
            fig.add_vline(x=integrated_data['Age'].mean(), line_dash="dash", 
                         annotation_text=f"Rata-rata: {integrated_data['Age'].mean():.1f}")
            fig.add_vline(x=integrated_data['Age'].median(), line_dash="dot", 
                         annotation_text=f"Median: {integrated_data['Age'].median():.1f}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Aktivitas per Kelompok Usia
        age_activity = integrated_data.groupby('age_group')[['friend_count', 'post_count', 'reactions_given', 'reactions_received']].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=age_activity.index, y=age_activity['friend_count'], 
                                mode='lines+markers', name='Rata-rata Teman', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=age_activity.index, y=age_activity['post_count'], 
                                mode='lines+markers', name='Rata-rata Post', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=age_activity.index, y=age_activity['reactions_given'], 
                                mode='lines+markers', name='Rata-rata Reaksi Diberikan', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=age_activity.index, y=age_activity['reactions_received'], 
                                mode='lines+markers', name='Rata-rata Reaksi Diterima', line=dict(width=3)))
        
        fig.update_layout(
            title="Pola Aktivitas per Kelompok Usia",
            xaxis_title="Kelompok Usia",
            yaxis_title="Rata-rata Aktivitas",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Performers
    elif selected_section == "🏆 Top Performers":
        st.markdown('<h2 class="sub-header">🏆 Top Performers</h2>', unsafe_allow_html=True)
        
        # Filter untuk top N
        top_n = st.slider("Tampilkan Top N pengguna:", 5, 20, 10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top Active Users
            top_active = integrated_data.nlargest(top_n, 'total_activity')
            fig = px.bar(
                top_active,
                x='total_activity',
                y=[f"{row['Name']} {row['Surname']}" for _, row in top_active.iterrows()],
                orientation='h',
                title=f"Top {top_n} Pengguna Paling Aktif",
                labels={'total_activity': 'Total Aktivitas', 'y': 'Pengguna'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top Posters
            top_posters = integrated_data.nlargest(top_n, 'post_count')
            fig = px.bar(
                top_posters,
                x='post_count',
                y=[f"{row['Name']} {row['Surname']}" for _, row in top_posters.iterrows()],
                orientation='h',
                title=f"Top {top_n} Content Creator",
                labels={'post_count': 'Jumlah Post', 'y': 'Pengguna'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("📋 Tabel Detail Top Performers")
        display_cols = ['Name', 'Surname', 'Age', 'friend_count', 'post_count', 'reactions_given', 'reactions_received', 'total_activity']
        top_detailed = integrated_data.nlargest(top_n, 'total_activity')[display_cols]
        st.dataframe(top_detailed, use_container_width=True)
    
    # Analisis Reaksi
    elif selected_section == "💝 Analisis Reaksi":
        st.markdown('<h2 class="sub-header">💝 Analisis Reaksi</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart reaksi
            reaction_counts = reactions_processed['Reaction Type'].value_counts()
            fig = px.pie(
                values=reaction_counts.values,
                names=reaction_counts.index,
                title="Proporsi Jenis Reaksi"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart reaksi
            fig = px.bar(
                x=reaction_counts.index,
                y=reaction_counts.values,
                title="Jumlah Reaksi per Jenis",
                labels={'x': 'Jenis Reaksi', 'y': 'Jumlah'},
                color=reaction_counts.values,
                color_continuous_scale='plasma'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Timeline reaksi
        reactions_timeline = reactions_processed.copy()
        reactions_timeline['date'] = pd.to_datetime(reactions_timeline['Reaction Date']).dt.date
        daily_reactions = reactions_timeline.groupby('date').size().reset_index(name='count')
        
        fig = px.line(
            daily_reactions,
            x='date',
            y='count',
            title="Timeline Aktivitas Reaksi Harian"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribusi Aktivitas
    elif selected_section == "📊 Distribusi Aktivitas":
        st.markdown('<h2 class="sub-header">📊 Distribusi Aktivitas</h2>', unsafe_allow_html=True)
        
        # Level aktivitas
        def categorize_activity(row):
            if row['total_activity'] == 0:
                return 'Tidak Aktif'
            elif row['total_activity'] <= 5:
                return 'Aktivitas Rendah'
            elif row['total_activity'] <= 15:
                return 'Aktivitas Sedang'
            elif row['total_activity'] <= 30:
                return 'Aktivitas Tinggi'
            else:
                return 'Sangat Aktif'
        
        integrated_data['activity_level'] = integrated_data.apply(categorize_activity, axis=1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart level aktivitas
            activity_counts = integrated_data['activity_level'].value_counts()
            fig = px.pie(
                values=activity_counts.values,
                names=activity_counts.index,
                title="Proporsi Level Aktivitas Pengguna"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Engagement ratio distribution
            posting_users = integrated_data[integrated_data['post_count'] > 0]
            fig = px.histogram(
                posting_users,
                x='engagement_ratio',
                title="Distribusi Engagement Ratio",
                labels={'engagement_ratio': 'Engagement Ratio', 'count': 'Jumlah Pengguna'},
                nbins=30
            )
            fig.add_vline(x=posting_users['engagement_ratio'].mean(), line_dash="dash",
                         annotation_text=f"Rata-rata: {posting_users['engagement_ratio'].mean():.2f}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Post type analysis
        if 'Post Type' in posts_processed.columns:
            post_type_counts = posts_processed['Post Type'].value_counts()
            fig = px.bar(
                x=post_type_counts.index,
                y=post_type_counts.values,
                title="Distribusi Jenis Postingan",
                labels={'x': 'Jenis Post', 'y': 'Jumlah'},
                color=post_type_counts.values,
                color_continuous_scale='viridis'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Analisis Korelasi
    elif selected_section == "🔗 Analisis Korelasi":
        st.markdown('<h2 class="sub-header">🔗 Analisis Korelasi</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot: Friends vs Posts
            fig = px.scatter(
                integrated_data,
                x='friend_count',
                y='post_count',
                color='total_activity',
                size='Age',
                title="Hubungan Jumlah Teman vs Postingan",
                labels={'friend_count': 'Jumlah Teman', 'post_count': 'Jumlah Post'},
                color_continuous_scale='viridis'
            )
            
            # Add trendline
            correlation = integrated_data['friend_count'].corr(integrated_data['post_count'])
            fig.add_annotation(
                text=f"Korelasi: {correlation:.3f}",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="yellow",
                bordercolor="black"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter plot: Age vs Activity
            fig = px.scatter(
                integrated_data,
                x='Age',
                y='total_activity',
                color='age_group',
                title="Hubungan Usia vs Total Aktivitas",
                labels={'Age': 'Usia', 'total_activity': 'Total Aktivitas'}
            )
            
            age_activity_corr = integrated_data['Age'].corr(integrated_data['total_activity'])
            fig.add_annotation(
                text=f"Korelasi: {age_activity_corr:.3f}",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="yellow",
                bordercolor="black"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        correlation_vars = ['Age', 'friend_count', 'post_count', 'reactions_given', 'reactions_received', 'total_activity']
        correlation_matrix = integrated_data[correlation_vars].corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Matriks Korelasi Antar Variabel",
            color_continuous_scale='RdBu'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights Mendalam
    elif selected_section == "🎯 Insights Mendalam":
        st.markdown('<h2 class="sub-header">🎯 Insights Mendalam</h2>', unsafe_allow_html=True)
        
        # Heatmap aktivitas per usia dan tahun
        activity_heatmap = integrated_data.groupby(['age_group', 'registration_year'])['total_activity'].mean().unstack()
        
        fig = px.imshow(
            activity_heatmap,
            text_auto=True,
            aspect="auto",
            title="Pola Aktivitas Berdasarkan Usia dan Tahun Bergabung",
            labels={'x': 'Tahun Registrasi', 'y': 'Kelompok Usia'},
            color_continuous_scale='YlOrRd'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("🔍 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Temuan Utama:**")
            
            # Most active age group
            most_active_age = integrated_data.groupby('age_group')['total_activity'].mean().idxmax()
            st.write(f"• Kelompok usia paling aktif: **{most_active_age}**")
            
            # Engagement insights
            posting_users = integrated_data[integrated_data['post_count'] > 0]
            high_engagement_pct = (posting_users['engagement_ratio'] > posting_users['engagement_ratio'].median()).mean() * 100
            st.write(f"• {high_engagement_pct:.1f}% content creator memiliki engagement di atas median")
            
            # Social correlation
            correlation = integrated_data['friend_count'].corr(integrated_data['post_count'])
            if correlation > 0.3:
                social_insight = "Pengguna yang lebih sosial cenderung lebih aktif posting"
            elif correlation < -0.3:
                social_insight = "Pengguna yang lebih sosial cenderung kurang aktif posting"
            else:
                social_insight = "Tidak ada hubungan kuat antara jumlah teman dan aktivitas posting"
            st.write(f"• {social_insight}")
        
        with col2:
            st.markdown("**🎯 Rekomendasi:**")
            
            # Top reaction type
            top_reaction = reactions_processed['Reaction Type'].value_counts().index[0]
            st.write(f"• Fokus pada konten yang mendorong reaksi '**{top_reaction}**'")
            
            # Activity distribution
            inactive_pct = (integrated_data['total_activity'] == 0).mean() * 100
            st.write(f"• {inactive_pct:.1f}% pengguna tidak aktif - peluang engagement")
            
            # Age targeting
            st.write(f"• Target kelompok usia **{most_active_age}** untuk fitur baru")
            
            # Posting frequency
            avg_posts = posting_users['post_count'].mean()
            st.write(f"• Rata-rata {avg_posts:.1f} post per active user - dorong konsistensi")
        
        # Interactive filters
        st.subheader("🎛️ Eksplorasi Interaktif")
        
        # Age filter
        age_range = st.slider(
            "Filter Rentang Usia:",
            int(integrated_data['Age'].min()),
            int(integrated_data['Age'].max()),
            (int(integrated_data['Age'].min()), int(integrated_data['Age'].max()))
        )
        
        # Activity filter
        activity_threshold = st.slider(
            "Minimal Total Aktivitas:",
            0,
            int(integrated_data['total_activity'].max()),
            0
        )
        
        # Apply filters
        filtered_data = integrated_data[
            (integrated_data['Age'] >= age_range[0]) &
            (integrated_data['Age'] <= age_range[1]) &
            (integrated_data['total_activity'] >= activity_threshold)
        ]
        
        # Show filtered statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👥 Pengguna Terfilter", len(filtered_data))
        with col2:
            st.metric("📊 Rata-rata Aktivitas", f"{filtered_data['total_activity'].mean():.1f}")
        with col3:
            st.metric("🎯 Rata-rata Engagement", f"{filtered_data['engagement_ratio'].mean():.2f}")
        
        # Filtered visualization
        if len(filtered_data) > 0:
            fig = px.scatter(
                filtered_data,
                x='friend_count',
                y='post_count',
                color='age_group',
                size='total_activity',
                title="Data Terfilter: Teman vs Post",
                hover_data=['Name', 'Surname', 'Age']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()