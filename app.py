import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter

import PyPDF2
import os

# Define the folder path
folder_path = 'kw_to_add'

def find_file_pairs(folder):
    files = os.listdir(folder)
    pdf_files = {os.path.splitext(file)[0] for file in files if file.endswith('.pdf')}
    csv_files = {os.path.splitext(file)[0] for file in files if file.endswith('.csv')}
    paired_files = pdf_files.intersection(csv_files)
    return paired_files

# Funkcja do wyodrębniania tekstu z pliku PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    
    # Zmienna przechowująca linię zawierającą 'Filterkriterien : Schichtdatum'
    filterkriterien_line = None
    previous_line = ""
    filtered_lines = []
    
    for line in text.split('\n'):
        line = line.strip()
        
        if line == 'Maschine':
            continue
        if 'Anzahl: Summen:' in line:
            continue
        if 'Filterkriterien : Schichtdatum' in line:
            if filterkriterien_line is None:
                filterkriterien_line = line
            continue
        if 'S-Schlüssel' in line:
            continue
        if 'Stillstand pro Abteilung und Maschine (Liste) Sortierung nach Grund' in line:
            line = line.replace('Stillstand pro Abteilung und Maschine (Liste) Sortierung nach Grund', '').strip()
            if line:
                filtered_lines.append(line)
            continue
        if 'www.syncos.info erstellt von' in line:
            parts = line.split('von', 2)
            if len(parts) > 2:
                line = parts[2].strip()
                if line:
                    filtered_lines.append(line)
            continue
        if previous_line and previous_line.split()[0].isdigit() and not line.split()[0].isdigit():
            # Przeniesienie linii z identyfikatorem maszyny do poprzedniej linii
            filtered_lines[-1] = f"{filtered_lines[-1]} {line}"
            continue

        filtered_lines.append(line)
        previous_line = line
    
    filtered_text = '\n'.join(filtered_lines)
    
    return filtered_text, filterkriterien_line

import re

def find_unique_machines(text):
    lines = text.split('\n')
    machine_pattern = re.compile(r'^\d{5}\s[\w\s/]+')
    unique_machines = set()

    for line in lines:
        if machine_pattern.match(line):
            unique_machines.add(line.strip())
    
    return list(unique_machines)

import pandas as pd
import re

def load_data_to_dataframe(text):
    lines = text.split('\n')
    data = []
    current_machine_id = None
    current_machine_name = None
    
    # Regex dla zdarzeń
    event_pattern = re.compile(r'^(\d+)\s([\w\s/-]+)\s(\d+)\s(\d{2}\.\d{2}\.\d{4}\s\d{2}:\d{2})\s(\d+)$')
    
    # Znajdź unikalne maszyny
    unique_machines = find_unique_machines(text)
    machine_pattern = re.compile(r'^\d{5}\s[\w\s/]+')
    
    for line in lines:
        line = line.strip()
        if machine_pattern.match(line):
            current_machine_id = line.split()[0]
            current_machine_name = ' '.join(line.split()[1:])
        elif current_machine_id and (event_match := event_pattern.match(line)):
            event_code = event_match.group(1)
            event_type = event_match.group(2)
            order_number = event_match.group(3)
            date_time = event_match.group(4)
            duration = event_match.group(5)
            data.append([current_machine_id, current_machine_name, event_code, event_type, order_number, pd.to_datetime(date_time, dayfirst=True), int(duration)])
    
    # Tworzenie DataFrame
    df = pd.DataFrame(data, columns=['Machine_ID', 'Machine_Name', 'Event_Code', 'Event_Type', 'Order_Number', 'Date_Time', 'Duration'])
    return df

# Funkcja do rozdzielenia kolumny 'Numer maszyny/Nazwa maszyny'
def split_machine_name(row):
    parts = row.split('/')
    if len(parts) == 3:
        return parts[0], parts[1]
    elif len(parts) == 2:
        return parts[0], parts[1]
    else:
        return row, None

# Funkcja do przetwarzania pojedynczego pliku CSV
def process_csv_file(filename):
    try:
        data = pd.read_csv(filename, encoding='latin-1', sep=';', usecols=['Kst./Maschine', 'Theoretische Schichtzeit ()'])
        print(f"File {filename} loaded successfully with Latin-1 encoding.")
    except UnicodeDecodeError as e:
        print(f"Failed to load file {filename} with Latin-1 encoding: {e}")
        return None

    # Zmiana nazw kolumn
    data.rename(columns={
        'Kst./Maschine': 'Numer maszyny/Nazwa maszyny',
        'Theoretische Schichtzeit ()': 'Theoretical_Time',
    }, inplace=True)

    # Rozdzielenie kolumny 'Numer maszyny/Nazwa maszyny'
    data[['Machine_ID', 'Machine_Name']] = data['Numer maszyny/Nazwa maszyny'].apply(split_machine_name).apply(pd.Series)
    data.drop(columns=['Numer maszyny/Nazwa maszyny'], inplace=True)

    # Ekstrakcja numeru KW
    kw_number = filename.split('KW')[1].replace('.csv', '')
    data['KW'] = int(kw_number)

    return data








st.set_page_config(layout="wide")








with st.expander('ADD NEW KW DATA'):

    data_2024 = pd.read_csv('2024.csv')

    # Check for file pairs in the specified folder
    file_pairs = find_file_pairs(folder_path)
    
    if file_pairs:
        st.write("Found the following file pairs:")
        selected_pair = st.selectbox("Select a file pair to process:", list(file_pairs))
        
        if st.button('Add Data'):
            with st.spinner(''):
                pdf_file_path = os.path.join(folder_path, f"{selected_pair}.pdf")
                csv_file_path = os.path.join(folder_path, f"{selected_pair}.csv")
                
                # Process the files using the existing function
                extracted_text, filterkriterien = extract_text_from_pdf(pdf_file_path)
                
                # Dataframe
                new_data_df = load_data_to_dataframe(extracted_text)

                # Dodaj kolumnę z numerem tygodnia kalendarzowego
                new_data_df['KW'] = new_data_df['Date_Time'].dt.isocalendar().week

                departments = pd.read_csv('lista_maszyn.csv')

                # Convert to the same type
                new_data_df['Machine_ID'] = new_data_df['Machine_ID'].astype(str)
                departments['Machine_ID'] = departments['Machine_ID'].astype(str)

                # Łączenie df z departments
                new_data_df = pd.merge(new_data_df, departments, on='Machine_ID', how='left')

                # Removing 'Pusty' Event Type
                new_data_df = new_data_df[new_data_df['Event_Type'] != 'Pusty']

                # Process theoretical times
                theoretical_times = process_csv_file(csv_file_path)

                theoretical_times = theoretical_times.drop(columns=['Machine_Name'])

                # Merge new data
                # Przyporządkowanie teoretycznego czasu z theoretical_times do data_2024
                # Wykonujemy lewe połączenie, aby zachować wszystkie rekordy z data_2024
                new_data_df = pd.merge(new_data_df, theoretical_times, on=['KW', 'Machine_ID'], how='left')

                # Format the Date_Time column to exclude seconds
                new_data_df['Date_Time'] = pd.to_datetime(new_data_df['Date_Time'])
                new_data_df['Date_Time'] = new_data_df['Date_Time'].dt.strftime('%m/%d/%Y %H:%M')
                
                # Concatenate data_2024 with new_data_df
                data_2024 = pd.concat([data_2024, new_data_df], ignore_index=True)

                st.dataframe(data_2024)
                
                # Drop duplicates
                before_dropping = len(data_2024)
                data_2024.drop_duplicates(subset=['Machine_Name', 'Date_Time', 'KW'], inplace=True)
                after_dropping = len(data_2024)
                dropped_duplicates = before_dropping - after_dropping

                data_2024.to_csv('2024.csv', index=False)

                # Display success message
                st.write(f"Number of dropped duplicates: {dropped_duplicates}")
                st.success("Data added and concatenated successfully!")


    else:
        st.write("No matching file pairs found.")









st.divider()














data_2024 = pd.read_csv('2024.csv')

# Convert the 'Machine_ID' column to string
data_2024['Machine_ID'] = data_2024['Machine_ID'].astype(str)

# Convert the 'Order_Number' column to string
data_2024['Order_Number'] = data_2024['Order_Number'].astype(str)

# Convert 'Duration' to hours
data_2024['Duration'] = round(data_2024['Duration']/60, 2)

# Convert 'Theoretical_Time' column to hours
data_2024['Theoretical_Time'] = round(data_2024['Theoretical_Time']/60, 2)

# Convert the 'Date_Time' column to datetime
data_2024['Date_Time'] = pd.to_datetime(data_2024['Date_Time'])

data_2024 = data_2024.sort_values(by='Date_Time')

unique_machine_names = data_2024['Machine_Name'].unique()

unique_machine_ids = data_2024['Machine_ID'].unique()

event_types = list(data_2024['Event_Type'].unique())





















with st.expander('INDIVIDUAL MACHINE STATISTICS'):

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('**Choose machine to show statistics**')
        choose_options = st.radio('', options=['Machine name', 'Machine number'], 
                                label_visibility='collapsed', horizontal=True)

        if choose_options == 'Machine number':
            chosen_machine = st.selectbox('', options=unique_machine_ids,
                                        label_visibility='collapsed')
        
        elif choose_options == 'Machine name':
            chosen_machine = st.selectbox('', options=unique_machine_names, 
                                        label_visibility='collapsed')


        
        col3, col4 = st.columns([1, 1])

        with col3:
            st.markdown('**Choose appropriate period**')
            
            period_option = st.radio('Period', options=['Yearly', 'Monthly', 'Weekly'],
                                    label_visibility='collapsed', horizontal=True)

        with col4:
            if period_option == 'Monthly':
                month_options = st.multiselect('Choose month(s)', options=range(1, 13))
            elif period_option == 'Weekly':
                week_options = st.multiselect('Choose week(s)', options=data_2024['KW'].unique())

        st.divider()

        filtered_data = data_2024

        if period_option == 'Monthly' and month_options:
            filtered_data = filtered_data[filtered_data['Date_Time'].dt.month.isin(month_options)]
        elif period_option == 'Weekly' and week_options:
            filtered_data = filtered_data[filtered_data['KW'].isin(week_options)]
        
        if choose_options == 'Machine name':
            filtered_data = filtered_data[filtered_data['Machine_Name'] == chosen_machine]
        elif choose_options == 'Machine number':
            filtered_data = filtered_data[filtered_data['Machine_ID'] == chosen_machine]

        options = st.radio('', options=['MTBF over KW', 'Kaplan-Meier Estimator'],
                           horizontal=True, label_visibility='collapsed')




        # Preparing data to calculate MTBF
        relevant_events = ['Problem mechaniczny', 'Problem elektryczny', 'Naprawa', 'Serwis']

        mtbf_events_data = filtered_data[filtered_data['Event_Type'].isin(relevant_events)]

        # MTTR
        mttr_data = mtbf_events_data.groupby(['Machine_Name', 'KW']).agg(
        Total_Duration=('Duration', 'sum'),  # Suma czasu trwania zdarzeń
        Event_Count=('Machine_Name', 'size'),
        Department=('Department', 'first')  # Liczba zdarzeń
        ).reset_index()

        mttr_data['MTTR'] = round(mttr_data['Total_Duration'] / mttr_data['Event_Count'], 2)

        # MTTF
        # Grupowanie danych po nazwie maszyny i tygodniu kalendarzowym (KW)
        mttf_data = mtbf_events_data.groupby(['Machine_Name', 'KW']).agg(
            Event_Count=('Machine_Name', 'size'),  # Liczba zdarzeń
            Total_Duration=('Duration', 'sum'),  # Suma czasu trwania zdarzeń
            Theoretical_Time=('Theoretical_Time', 'first'),
            Department=('Department', 'first')  # Suma teoretycznych czasów pracy
        ).reset_index()

        # Obliczanie MTTF jako różnica między teoretycznym czasem pracy a czasem trwania zdarzeń, podzielona przez liczbę zdarzeń
        mttf_data['MTTF'] = round((mttf_data['Theoretical_Time'] - mttf_data['Total_Duration']) / mttf_data['Event_Count'], 2)

        # MTBF

        mtbf_data = pd.merge(mttr_data, mttf_data, on=['Machine_Name', 'KW'], how='inner')

        # Obliczenie MTBF jako suma MTTR i MTTF
        mtbf_data['MTBF'] = mtbf_data['MTTR'] + mtbf_data['MTTF']

        # Catch errors
        errors = mtbf_data[mtbf_data['Theoretical_Time'] < mtbf_data['Total_Duration_x']]

        # Filter out rows where Theoretical_Time is less than Total_Duration
        mtbf_data = mtbf_data[mtbf_data['Theoretical_Time'] >= mtbf_data['Total_Duration_x']]

        # Plot MTBF data
        avg_mtbf = mtbf_data['MTBF'].mean()
        fig_mtbf = px.line(mtbf_data, x='KW', y='MTBF', title='MTBF over KW', markers=True,
                        hover_data={'MTTR': True, 'MTTF': True, 'MTBF': True})
        
        if options == 'MTBF over KW':

            # Add the average MTBF as a dashed line
            fig_mtbf.add_shape(
                type="line",
                x0=mtbf_data['KW'].min(), x1=mtbf_data['KW'].max(),
                y0=avg_mtbf, y1=avg_mtbf,
                line=dict(color="Green", width=2, dash="dash"),
            )

            # Add annotation for the average MTBF
            fig_mtbf.add_annotation(
                x=mtbf_data['KW'].max(),
                y=avg_mtbf,
                text=f"Average MTBF: {avg_mtbf:.2f}",
                showarrow=False,
                yshift=10,
                font=dict(color="Green")
            )
            
            st.plotly_chart(fig_mtbf, use_container_width=True)

        elif options == 'Kaplan-Meier Estimator':

            # Preparing data to calculate MTBF
            relevant_events = ['Problem mechaniczny', 'Problem elektryczny', 'Naprawa', 'Serwis']

            mtbf_events_data = filtered_data[filtered_data['Event_Type'].isin(relevant_events)]

            survival_data = mtbf_events_data

            # Przetwarzanie daty i czasu, sortowanie i obliczanie interwałów
            survival_data['Date_Time'] = pd.to_datetime(survival_data['Date_Time'])
            data_sorted = survival_data.sort_values(['Machine_ID', 'Date_Time'])
            data_sorted['Time_to_Next_Failure'] = data_sorted.groupby('Machine_ID')['Date_Time'].diff().shift(-1)
            data_sorted['Time_to_Next_Failure'] = data_sorted['Time_to_Next_Failure'].dt.total_seconds() / 3600
            data_for_analysis = data_sorted.dropna(subset=['Time_to_Next_Failure'])

            if not data_for_analysis.empty:
                # Tworzenie obiektu Kaplan-MeierFitter i dopasowanie modelu
                kmf = KaplanMeierFitter()
                kmf.fit(data_for_analysis['Time_to_Next_Failure'], event_observed=[1]*len(data_for_analysis))

                # Przygotowanie danych do wykresu
                surv_func = kmf.survival_function_
                surv_func.reset_index(inplace=True)
                surv_func.columns = ['Time_in_Hours', 'Survival_Probability']

                # Pasma przedziałów ufności
                confidence_intervals = kmf.confidence_interval_
                confidence_intervals.reset_index(inplace=True)
                confidence_intervals.columns = ['Time_in_Hours', 'CI_lower', 'CI_upper']

                # Tworzenie interaktywnego wykresu za pomocą Plotly
                fig = go.Figure([
                    go.Scatter(
                        name='Survival Curve',
                        x=surv_func['Time_in_Hours'],
                        y=surv_func['Survival_Probability'],
                        mode='lines',
                        line=dict(color='#2980cc'),
                        hoverinfo='text',
                        hovertext=[
                            f"Time: {time} hours<br>Survival Probability: {prob:.2f}"
                            for time, prob in zip(surv_func['Time_in_Hours'], surv_func['Survival_Probability'])
                        ],
                        showlegend=False
                    ),
                    go.Scatter(
                        name='Upper Bound',
                        x=confidence_intervals['Time_in_Hours'],
                        y=confidence_intervals['CI_upper'],
                        mode='lines',
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='none'
                    ),
                    go.Scatter(
                        name='Lower Bound',
                        x=confidence_intervals['Time_in_Hours'],
                        y=confidence_intervals['CI_lower'],
                        mode='lines',
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        fillcolor='rgba(0, 191, 255, 0.1)',
                        fill='tonexty',
                        showlegend=False,
                        hoverinfo='none'
                    )
                ])

                fig.update_layout(
                    title='Survival Function - Time to Next Machine Failure (Kaplan-Meier Estimator)',
                    xaxis_title='Time in Hours',
                    yaxis_title='Survival Probability (No Failure)',
                    template='plotly_white'
                )

                # Wyświetlanie wykresu w Streamlit
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No sufficient data available to analyze.")

    st.divider()



    with col2:


        # Preparing data to calculate MTBF
        relevant_events = ['Problem mechaniczny', 'Problem elektryczny', 'Naprawa', 'Serwis']

        mtbf_events_data = filtered_data[filtered_data['Event_Type'].isin(relevant_events)]

        st.markdown('**MTBF Data**')
        st.dataframe(mtbf_data[['Machine_Name',
                                'KW', 
                                'Total_Duration_x', 
                                'Event_Count_x', 
                                'Department_x',
                                'Theoretical_Time',
                                'MTTR',
                                'MTTF',
                                'MTBF'
                                ]], use_container_width=True, hide_index=True, height=200)

        st.markdown('**Syncos Status Data**')
        data_options = st.radio('', options = ['All Events', 'MTBF Events'], horizontal=True, label_visibility='collapsed')

        if data_options == 'MTBF Events':
            st.dataframe(mtbf_events_data, use_container_width=True, hide_index=True, height=350)

        if data_options == 'All Events':
            st.dataframe(filtered_data, use_container_width=True, hide_index=True, height=350)
        


    if not errors.empty:
        st.markdown('**Caught errors**')
        st.dataframe(errors, use_container_width=True)
    else:
        st.markdown('**There is no errors in data!**')

    st.divider()
    
    # Create a bar plot with Plotly
    if not filtered_data.empty:

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('**Event Duration by Type**')
            event_duration = filtered_data.groupby('Event_Type')['Duration'].sum().reset_index()
            event_duration = event_duration.sort_values(by='Duration', ascending=True)  # Sort the bars in ascending order
            fig = px.bar(event_duration, x='Duration', 
                        y='Event_Type', 
                        orientation='h')

            # Update layout to increase height, add grid, and remove y-axis title
            fig.update_layout(
                height=550,
                yaxis_title=None,
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:

            st.markdown('**Choose event type(s)**')
            # Get sorted event types based on duration
            sorted_event_types = event_duration['Event_Type'].tolist()[::-1]
            event_type_options = st.multiselect('', options=sorted_event_types, default= sorted_event_types[:3], label_visibility='collapsed')

            # Create a line plot for the selected event types with transparent bars
            if event_type_options:
                line_data = filtered_data[filtered_data['Event_Type'].isin(event_type_options)]
                
                # Create the scatter plot with variable marker size
                fig_scatter = px.scatter(
                    line_data, 
                    x='Date_Time', 
                    y='Duration', 
                    color='Event_Type', 
                    size='Duration',  # Set marker size based on Duration
                    title="Duration over Time by Event Type", 

                )

                # Create the bar plot with transparent bars
                fig_bar = px.bar(
                    line_data, 
                    x='Date_Time', 
                    y='Duration', 
                    color='Event_Type', 
                    opacity=0.3,
                    barmode='overlay', 

                )

                # Combine scatter plot and bar plot
                fig_combined = go.Figure(data=fig_scatter.data + fig_bar.data)
                
                # Update layout
                fig_combined.update_layout(title="Duration over Time by Event Type")
                
                st.plotly_chart(fig_combined, use_container_width=True)

st.divider()


























with st.expander('DEPARTMENT STATISTICS'):
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('**Choose department to show statistics**')
        chosen_department = st.selectbox('', options=data_2024['Department'].unique(), 
                                         label_visibility='collapsed')
        
        col3, col4 = st.columns([1, 1])

        with col3:
            st.markdown('**Choose appropriate period**')
            period_option = st.radio('Period', options=['Yearly', 'Monthly', 'Weekly'],
                                     label_visibility='collapsed', horizontal=True, key = '2')

        with col4:
            if period_option == 'Monthly':
                month_options = st.multiselect('Choose month(s)', options=range(1, 13), key = '3')
            elif period_option == 'Weekly':
                week_options = st.multiselect('Choose week(s)', options=data_2024['KW'].unique(), key = '4')

        st.divider()

        filtered_data = data_2024

        if period_option == 'Monthly' and month_options:
            filtered_data = filtered_data[filtered_data['Date_Time'].dt.month.isin(month_options)]
        elif period_option == 'Weekly' and week_options:
            filtered_data = filtered_data[filtered_data['KW'].isin(week_options)]
        
        filtered_data = filtered_data[filtered_data['Department'] == chosen_department]

        # Preparing data to calculate MTBF
        relevant_events = ['Problem mechaniczny', 'Problem elektryczny', 'Naprawa', 'Serwis']

        mtbf_events_data = filtered_data[filtered_data['Event_Type'].isin(relevant_events)]

        # MTTR
        mttr_data = mtbf_events_data.groupby(['Machine_Name', 'KW']).agg(
        Total_Duration=('Duration', 'sum'),  # Suma czasu trwania zdarzeń
        Event_Count=('Machine_Name', 'size'),
        Department=('Department', 'first')  # Liczba zdarzeń
        ).reset_index()

        mttr_data['MTTR'] = round(mttr_data['Total_Duration'] / mttr_data['Event_Count'], 2)

        # MTTF
        # Grupowanie danych po nazwie maszyny i tygodniu kalendarzowym (KW)
        mttf_data = mtbf_events_data.groupby(['Machine_Name', 'KW']).agg(
            Event_Count=('Machine_Name', 'size'),  # Liczba zdarzeń
            Total_Duration=('Duration', 'sum'),  # Suma czasu trwania zdarzeń
            Theoretical_Time=('Theoretical_Time', 'first'),
            Department=('Department', 'first')  # Suma teoretycznych czasów pracy
        ).reset_index()

        # Obliczanie MTTF jako różnica między teoretycznym czasem pracy a czasem trwania zdarzeń, podzielona przez liczbę zdarzeń
        mttf_data['MTTF'] = round((mttf_data['Theoretical_Time'] - mttf_data['Total_Duration']) / mttf_data['Event_Count'], 2)

        # MTBF
        mtbf_data = pd.merge(mttr_data, mttf_data, on=['Machine_Name', 'KW'], how='inner')

        # Obliczenie MTBF jako suma MTTR i MTTF
        mtbf_data['MTBF'] = mtbf_data['MTTR'] + mtbf_data['MTTF']

        # Catch errors
        errors = mtbf_data[mtbf_data['Theoretical_Time'] < mtbf_data['Total_Duration_x']]

        # Filter out rows where Theoretical_Time is less than Total_Duration
        mtbf_data = mtbf_data[mtbf_data['Theoretical_Time'] >= mtbf_data['Total_Duration_x']]

        st.markdown('**Choose MTBF view**')
        mtbf_view_option = st.radio('', options=['Average MTBF per Machine', 'Overall MTBF per KW', 'Kaplan-Meier Estimator'], 
                                    label_visibility='collapsed', horizontal=True, key='5')

        if mtbf_view_option == 'Average MTBF per Machine':
            # Group by 'Machine_Name' and calculate the mean MTBF
            average_mtbf_data = mtbf_data.groupby('Machine_Name').agg(
                Average_MTBF=('MTBF', 'mean')
            ).reset_index()

            # Sort the average MTBF data
            average_mtbf_data = average_mtbf_data.sort_values(by='Average_MTBF', ascending=False)

            # Plot the average MTBF data
            fig_avg_mtbf = px.bar(average_mtbf_data, x='Machine_Name', y='Average_MTBF', 
                                title='Average MTBF per Machine', labels={'Average_MTBF': 'Average MTBF (hours)'})
            st.plotly_chart(fig_avg_mtbf, use_container_width=True)
        
        elif mtbf_view_option == 'Overall MTBF per KW':
            # Calculate the average MTBF per KW across all machines
            avg_mtbf_kw = mtbf_data.groupby('KW').agg(
                Average_MTBF=('MTBF', 'mean')
            ).reset_index()

            # Calculate the overall average of these averages
            overall_avg_mtbf = avg_mtbf_kw['Average_MTBF'].mean()

            # Allow users to switch between line chart and bar chart
            chart_type = st.radio("Select Chart Type:", ['Line Chart', 'Bar Chart'], 
                                  horizontal=True, key='h')

            # Configure the plot according to the selected chart type
            if chart_type == 'Line Chart':
                fig = px.line(avg_mtbf_kw, x='KW', y='Average_MTBF',
                            title='Overall MTBF per KW',
                            labels={'Average_MTBF': 'Average MTBF (hours)', 'KW': 'KW'},
                            markers=True)
            else:
                fig = px.bar(avg_mtbf_kw, x='KW', y='Average_MTBF',
                            title='Overall MTBF per KW',
                            labels={'Average_MTBF': 'Average MTBF (hours)', 'KW': 'KW'},
                            color_discrete_sequence=['lightblue'])

            # Add a horizontal line for the overall average MTBF
            fig.add_shape(
                type="line",
                x0=avg_mtbf_kw['KW'].min(), x1=avg_mtbf_kw['KW'].max(),
                y0=overall_avg_mtbf, y1=overall_avg_mtbf,
                line=dict(color="Green", width=2, dash="dash"),
            )

            # Add annotation for the overall average MTBF
            fig.add_annotation(
                x=avg_mtbf_kw['KW'].max(),
                y=overall_avg_mtbf,
                text=f"Overall Average MTBF: {overall_avg_mtbf:.2f} hours",
                showarrow=False,
                yshift=10,
                font=dict(color="Green")
            )

            # Display the selected chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)

        if mtbf_view_option == 'Kaplan-Meier Estimator':
                # Preparing data to calculate MTBF
                relevant_events = ['Problem mechaniczny', 'Problem elektryczny', 'Naprawa', 'Serwis']

                mtbf_events_data = filtered_data[filtered_data['Event_Type'].isin(relevant_events)]

                survival_data = mtbf_events_data

                # Przetwarzanie daty i czasu, sortowanie i obliczanie interwałów
                survival_data['Date_Time'] = pd.to_datetime(survival_data['Date_Time'])
                data_sorted = survival_data.sort_values(['Machine_ID', 'Date_Time'])
                data_sorted['Time_to_Next_Failure'] = data_sorted.groupby('Machine_ID')['Date_Time'].diff().shift(-1)
                data_sorted['Time_to_Next_Failure'] = data_sorted['Time_to_Next_Failure'].dt.total_seconds() / 3600
                data_for_analysis = data_sorted.dropna(subset=['Time_to_Next_Failure'])

                # Initialize figure
                fig = go.Figure()

                # Loop through each machine
                for machine in data_for_analysis['Machine_Name'].unique():
                    machine_data = data_for_analysis[data_for_analysis['Machine_Name'] == machine]

                    if not machine_data.empty:
                        # Fitting Kaplan-Meier model
                        kmf = KaplanMeierFitter()
                        kmf.fit(machine_data['Time_to_Next_Failure'], event_observed=[1] * len(machine_data))

                        # Preparing survival function data
                        surv_func = kmf.survival_function_
                        surv_func.reset_index(inplace=True)
                        surv_func.columns = ['Time_in_Hours', 'Survival_Probability']

                        # Adding survival curve to the plot
                        fig.add_trace(go.Scatter(
                            name=f'{machine}',
                            x=surv_func['Time_in_Hours'],
                            y=surv_func['Survival_Probability'],
                            mode='lines',
                            line=dict(width=2),
                            hoverinfo='text',
                            hovertext=[
                                f"Machine: {machine}<br>Time: {time} hours<br>Survival Probability: {prob:.2f}"
                                for time, prob in zip(surv_func['Time_in_Hours'], surv_func['Survival_Probability'])
                            ]
                        ))

                # Update plot layout
                fig.update_layout(
                    title='Survival Functions for All Machines (Kaplan-Meier Estimator)',
                    xaxis_title='Time in Hours',
                    yaxis_title='Survival Probability (No Failure)',
                    legend_title="Machine Name",
                    template='plotly_white'
                )

                # Displaying the plot in Streamlit
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Preparing data to calculate MTBF
        relevant_events = ['Problem mechaniczny', 'Problem elektryczny', 'Naprawa', 'Serwis']

        mtbf_events_data = filtered_data[filtered_data['Event_Type'].isin(relevant_events)]

        st.markdown('**MTBF Data**')
        st.dataframe(mtbf_data[['Machine_Name',
                                'KW', 
                                'Total_Duration_x', 
                                'Event_Count_x', 
                                'Department_x',
                                'Theoretical_Time',
                                'MTTR',
                                'MTTF',
                                'MTBF'
                                ]], use_container_width=True, hide_index=True, height=200)
        
        st.markdown('**Syncos Status Data**')
        data_options = st.radio('', options = ['All Events', 'MTBF Events'], horizontal=True, label_visibility='collapsed', key='d')

        if data_options == 'MTBF Events':
            st.dataframe(mtbf_events_data, use_container_width=True, hide_index=True, height=350)

        if data_options == 'All Events':
            st.dataframe(filtered_data, use_container_width=True, hide_index=True, height=350)
        
    st.divider()

    if not errors.empty:
        st.markdown('**Caught errors**')
        st.dataframe(errors, use_container_width=True, height=300)
    else:
        st.markdown('**There is no errors in data!**')

st.divider()



























with st.expander('COMPANY STATISTICS'):
    col1, col2 = st.columns([1, 2])

    with col1:

        col3, col4 = st.columns([1, 1])

        with col3:
            st.markdown('**Choose appropriate period**')
            period_option = st.radio('Period', options=['Yearly', 'Monthly', 'Weekly'],
                                     label_visibility='collapsed', horizontal=True, key = '2b')

        with col4:
            if period_option == 'Monthly':
                month_options = st.multiselect('Choose month(s)', options=range(1, 13), key = '3b')
            elif period_option == 'Weekly':
                week_options = st.multiselect('Choose week(s)', options=data_2024['KW'].unique(), key = '4b')

        st.divider()

        filtered_data = data_2024

        if period_option == 'Monthly' and month_options:
            filtered_data = filtered_data[filtered_data['Date_Time'].dt.month.isin(month_options)]
        elif period_option == 'Weekly' and week_options:
            filtered_data = filtered_data[filtered_data['KW'].isin(week_options)]

        # Preparing data to calculate MTBF
        relevant_events = ['Problem mechaniczny', 'Problem elektryczny', 'Naprawa', 'Serwis']

        mtbf_events_data = filtered_data[filtered_data['Event_Type'].isin(relevant_events)]

        # MTTR
        mttr_data = mtbf_events_data.groupby(['Machine_Name', 'KW']).agg(
        Total_Duration=('Duration', 'sum'),  # Suma czasu trwania zdarzeń
        Event_Count=('Machine_Name', 'size'),
        Department=('Department', 'first')  # Liczba zdarzeń
        ).reset_index()

        mttr_data['MTTR'] = round(mttr_data['Total_Duration'] / mttr_data['Event_Count'], 2)

        # MTTF
        # Grupowanie danych po nazwie maszyny i tygodniu kalendarzowym (KW)
        mttf_data = mtbf_events_data.groupby(['Machine_Name', 'KW']).agg(
            Event_Count=('Machine_Name', 'size'),  # Liczba zdarzeń
            Total_Duration=('Duration', 'sum'),  # Suma czasu trwania zdarzeń
            Theoretical_Time=('Theoretical_Time', 'first'),
            Department=('Department', 'first')  # Suma teoretycznych czasów pracy
        ).reset_index()

        # Obliczanie MTTF jako różnica między teoretycznym czasem pracy a czasem trwania zdarzeń, podzielona przez liczbę zdarzeń
        mttf_data['MTTF'] = round((mttf_data['Theoretical_Time'] - mttf_data['Total_Duration']) / mttf_data['Event_Count'], 2)

        # MTBF
        mtbf_data = pd.merge(mttr_data, mttf_data, on=['Machine_Name', 'KW'], how='inner')

        # Obliczenie MTBF jako suma MTTR i MTTF
        mtbf_data['MTBF'] = mtbf_data['MTTR'] + mtbf_data['MTTF']

        # Catch errors
        errors = mtbf_data[mtbf_data['Theoretical_Time'] < mtbf_data['Total_Duration_x']]

        # Filter out rows where Theoretical_Time is less than Total_Duration
        mtbf_data = mtbf_data[mtbf_data['Theoretical_Time'] >= mtbf_data['Total_Duration_x']]

        st.markdown('**Choose MTBF view**')
        mtbf_view_option = st.radio('', options=['Average MTBF per Department', 'Overall MTBF per KW'], 
                                    label_visibility='collapsed', horizontal=True, key='5b')

        if mtbf_view_option == 'Average MTBF per Department':
            # Group by 'Machine_Name' and calculate the mean MTBF
            average_mtbf_data = mtbf_data.groupby('Department_x').agg(
                Average_MTBF=('MTBF', 'mean')
            ).reset_index()

            # Sort the average MTBF data
            average_mtbf_data = average_mtbf_data.sort_values(by='Average_MTBF', ascending=False)

            # Plot the average MTBF data
            fig_avg_mtbf = px.bar(average_mtbf_data, x='Department_x', y='Average_MTBF', 
                                title='Average MTBF per Department', labels={'Average_MTBF': 'Average MTBF (hours)'})
            st.plotly_chart(fig_avg_mtbf, use_container_width=True)
        
        elif mtbf_view_option == 'Overall MTBF per KW':
            # Calculate the average MTBF per KW across all machines
            avg_mtbf_kw = mtbf_data.groupby('KW').agg(
                Average_MTBF=('MTBF', 'mean')
            ).reset_index()

            # Calculate the overall average of these averages
            overall_avg_mtbf = avg_mtbf_kw['Average_MTBF'].mean()

            # Add a radio button to select the chart type
            chart_type = st.radio("Select Chart Type:", ['Line Chart', 'Bar Chart'], horizontal=True)

            # Initialize the figure based on selected chart type
            if chart_type == 'Line Chart':
                fig = px.line(avg_mtbf_kw, x='KW', y='Average_MTBF', 
                            title='Overall MTBF per KW',
                            labels={'Average_MTBF': 'Average MTBF (hours)', 'KW': 'KW'},
                            markers=True)
            else:
                fig = px.bar(avg_mtbf_kw, x='KW', y='Average_MTBF', 
                            title='Overall MTBF per KW',
                            labels={'Average_MTBF': 'Average MTBF (hours)', 'KW': 'KW'},
                            color_discrete_sequence=['lightblue'])  # Setting bar color

            # Add a horizontal line for the overall average MTBF
            fig.add_shape(
                type="line",
                x0=avg_mtbf_kw['KW'].min(), x1=avg_mtbf_kw['KW'].max(),
                y0=overall_avg_mtbf, y1=overall_avg_mtbf,
                line=dict(color="Green", width=2, dash="dash"),
            )

            # Add annotation for the overall average MTBF
            fig.add_annotation(
                x=avg_mtbf_kw['KW'].max(),
                y=overall_avg_mtbf,
                text=f"Overall Average MTBF: {overall_avg_mtbf:.2f}",
                showarrow=False,
                yshift=10,
                font=dict(color="Green")
            )

            # Display the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)


    with col2:
        # Preparing data to calculate MTBF
        relevant_events = ['Problem mechaniczny', 'Problem elektryczny', 'Naprawa', 'Serwis']

        mtbf_events_data = filtered_data[filtered_data['Event_Type'].isin(relevant_events)]

        st.markdown('**MTBF Data**')
        # Agregacja mtbf_data
        aggregated_mtbf_data = mtbf_data.groupby('Machine_Name').agg({
            'MTTR': 'mean',
            'MTTF': 'mean',
            'MTBF': 'mean'
        }).reset_index()

        # Wyświetlenie zgrupowanych danych
        st.dataframe(aggregated_mtbf_data[['Machine_Name', 'MTTR', 'MTTF', 'MTBF']],
                    use_container_width=True, hide_index=True, height=300)
        
        st.markdown('**Syncos Status Data**')
        data_options = st.radio('', options = ['All Events', 'MTBF Events'], horizontal=True, label_visibility='collapsed', key='c')

        if data_options == 'MTBF Events':
            st.dataframe(mtbf_events_data, use_container_width=True, hide_index=True, height=300)

        if data_options == 'All Events':
            st.dataframe(filtered_data, use_container_width=True, hide_index=True, height=300)
        
        
        
    if not errors.empty:
        st.markdown('**Caught errors**')
        st.dataframe(errors, use_container_width=True, height=200)


    # Create a bar plot with Plotly
    if not filtered_data.empty:


        event_duration = filtered_data.groupby('Event_Type')['Duration'].sum().reset_index()
        event_duration = event_duration.sort_values(by='Duration', ascending=True)  # Sort the bars in ascending order
        fig = px.bar(event_duration, x='Duration', 
                    y='Event_Type', title="Event Duration by Type", 
                    orientation='h')

        # Update layout to increase height, add grid, and remove y-axis title
        fig.update_layout(
            height=800,
            yaxis_title=None,
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )

        st.plotly_chart(fig, use_container_width=True)









