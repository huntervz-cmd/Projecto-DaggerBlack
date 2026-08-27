import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Sistema Predictivo de Rendimiento Académico",
    page_icon="🎓",
    layout="wide"
)

# --- 2. CONTROL DE ESTADO DE SESIÓN (AUTENTICACIÓN) ---
if 'autenticado' not in st.session_state:
    st.session_state['autenticado'] = False
if 'usuario_actual' not in st.session_state:
    st.session_state['usuario_actual'] = ""

# Base de datos simulada de profesores
USUARIOS_DB = {
    "profesor@unellez.edu.ve": "admin123"
}

# --- 3. MÓDULO DE LOGIN Y RECUPERACIÓN ---
def pantalla_login():
    st.title("🎓 Acceso al Sistema Predictivo Académico")
    st.caption("Prototipo de interacción docente con Inteligencia Artificial")
    
    col_cen, _ = st.columns([2, 1])
    
    with col_cen:
        tab_login, tab_recuperar = st.tabs(["🔑 Iniciar Sesión", "📧 Recuperar Contraseña"])
        
        with tab_login:
            st.subheader("Ingreso de Docentes")
            correo = st.text_input("Correo Electrónico", value="profesor@unellez.edu.ve")
            clave = st.text_input("Contraseña", type="password", value="admin123")
            
            if st.button("Ingresar al Sistema", type="primary"):
                if correo in USUARIOS_DB and USUARIOS_DB[correo] == clave:
                    st.session_state['autenticado'] = True
                    st.session_state['usuario_actual'] = correo
                    st.success("¡Acceso concedido! Cargando panel...")
                    st.rerun()
                else:
                    st.error("Credenciales incorrectas. Verifique correo o contraseña.")
        
        with tab_recuperar:
            st.subheader("Recuperación de Acceso")
            st.write("Ingrese su correo para recibir un token de restablecimiento.")
            correo_rec = st.text_input("Correo de Recuperación")
            
            if st.button("Enviar Token de Recuperación"):
                if correo_rec in USUARIOS_DB:
                    st.info("✉️ Se ha enviado un token de verificación (Simulado: `TK-984210`) a su correo electrónico.")
                else:
                    st.warning("El correo no se encuentra registrado en el sistema.")

# --- 4. MODELO DE MACHINE LEARNING OPTIMIZADO ---
@st.cache_resource
def entrenar_modelo():
    np.random.seed(42)
    n = 100
    asistencia = np.random.uniform(40, 100, n)
    tareas = np.random.uniform(5, 20, n)
    rend_previo = np.random.uniform(5, 20, n)
    salud = np.random.choice([0, 1], n)
    socioecon = np.random.uniform(1.0, 5.0, n)
    sociodem = np.random.choice([1, 2, 3], n)
    
    # Etiquetado lógico de riesgo
    riesgo = np.where((asistencia < 75) | (tareas < 10) | (rend_previo < 10), 1, 0)
    
    df_train = pd.DataFrame({
        'asistencia': asistencia.astype(float),
        'tareas': tareas.astype(float),
        'rendimiento_previo': rend_previo.astype(float),
        'situacion_medica': salud.astype(int),
        'socioeconomico': socioecon.astype(float),
        'sociodemografico': sociodem.astype(int)
    })
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(df_train, riesgo)
    return model

# --- 5. INTERFAZ PRINCIPAL DEL DOCENTE ---
def pantalla_docente():
    st.sidebar.title("📌 Panel de Control")
    st.sidebar.write(f"**Usuario:** {st.session_state['usuario_actual']}")
    
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state['autenticado'] = False
        st.rerun()

    st.title("📊 Panel de Interacción Predictiva y Diagnóstico Estudiantil")
    st.markdown("---")
    
    modelo = entrenar_modelo()
    
    col_input, col_results = st.columns([1, 1.2])
    
    with col_input:
        st.subheader("📝 Parámetros del Estudiante a Evaluar")
        
        nombre_est = st.text_input("Nombre del Estudiante", "Carlos Pérez")
        asistencia = st.slider("Porcentaje de Asistencia (%)", 0.0, 100.0, 68.0)
        tareas = st.slider("Promedio de Tareas (0-20)", 0.0, 20.0, 9.5)
        rend_previo = st.slider("Rendimiento Académico Previo (0-20)", 0.0, 20.0, 11.0)
        
        col_med, col_soc = st.columns(2)
        with col_med:
            situacion_medica = st.selectbox("Situación Médica", [0, 1], format_func=lambda x: "Sí (Registrada)" if x == 1 else "No (Ninguna)")
        with col_soc:
            socioeconomico = st.slider("Nivel Socioeconómico (1-5)", 1.0, 5.0, 2.5)
            
        sociodemografico = st.selectbox("Zona Sociodemográfica", [1, 2, 3], format_func=lambda x: {1: "Urbana", 2: "Suburbana", 3: "Rural"}[x])

    with col_results:
        st.subheader("🎯 Resultado del Diagnóstico de ML")
        
        # Mapeo estructurado para evitar discrepancias de dtypes en Scikit-Learn
        input_data = pd.DataFrame([{
            'asistencia': float(asistencia),
            'tareas': float(tareas),
            'rendimiento_previo': float(rend_previo),
            'situacion_medica': int(situacion_medica),
            'socioeconomico': float(socioeconomico),
            'sociodemografico': int(sociodemografico)
        }])
        
        probabilidad = modelo.predict_proba(input_data)[0][1] * 100
        prediccion = modelo.predict(input_data)[0]
        
        st.metric(label="Probabilidad de Deserción / Bajo Rendimiento", value=f"{probabilidad:.1f}%")
        
        if prediccion == 1 or asistencia < 75:
            st.error(f"🚨 **ALERTA CRÍTICA:** El estudiante **{nombre_est}** ha sido clasificado **EN RIESGO DEFICIENTE**.")
            st.warning("⚠️ **Recomendación de Intervención:**\n- Programar tutoría académica inmediata.\n- Verificar inasistencias (<75%).\n- Contactar al departamento de bienestar estudiantil.")
        else:
            st.success(f"✅ **ESTADO REGULAR:** El estudiante **{nombre_est}** presenta un rendimiento y asistencia dentro de los parámetros esperados.")

    # --- ANÁLISIS EXPLORATORIO DE DATOS (EDA) ---
    st.markdown("---")
    st.subheader("📈 Análisis Exploratorio de Datos (EDA) - Matriz de Correlación")
    st.caption("Gráfico generado con Matplotlib y Seaborn para apoyar al docente a identificar correlaciones clave.")
    
    np.random.seed(42)
    df_eda = pd.DataFrame({
        'Asistencia': np.random.uniform(50, 100, 50),
        'Tareas': np.random.uniform(8, 20, 50),
        'Rend_Previo': np.random.uniform(7, 20, 50),
        'Salud': np.random.choice([0, 1], 50),
        'Socioeconomico': np.random.uniform(1, 5, 50),
        'Riesgo_ML': np.random.choice([0, 1], 50)
    })
    
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(df_eda.corr(numeric_only=True), annot=True, cmap='RdYlGn_r', fmt=".2f", ax=ax, cbar=False)
    st.pyplot(fig)

# --- 6. CONTROLADOR DE FLUJO ---
if not st.session_state['autenticado']:
    pantalla_login()
else:
    pantalla_docente()
