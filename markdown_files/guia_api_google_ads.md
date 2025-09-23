# Guía para Configurar Cuenta de Prueba de Google Ads API

## Requisitos Previos
1. Tener una cuenta de Google (Gmail)
2. Acceso a [Google Ads](https://ads.google.com)
3. Navegador web actualizado (Chrome recomendado)

## Paso 1: Crear Cuenta de Prueba
1. Visita el [Administrador de Cuentas de Prueba](https://ads.google.com/aw/apitest/signup)
2. Haz clic en "Crear cuenta de prueba"
3. Selecciona tu país y zona horaria
4. Elige el tipo de cuenta:
   - **Cuenta de búsqueda**: Para pruebas generales
   - **Cuenta de display**: Para anuncios gráficos
   - **Cuenta de video**: Para YouTube Ads
   - **Cuenta de compras**: Para Google Shopping
5. Haz clic en "Crear cuenta"

## Paso 2: Obtener Customer ID
1. Inicia sesión en tu [consola de Google Ads](https://ads.google.com)
2. En la esquina superior derecha, haz clic en el icono de tu cuenta
3. Selecciona "Configuración de la cuenta"
4. En "Información de la cuenta", busca tu "ID de cliente"
   - Formato: `123-456-7890`
5. Copia este ID - lo necesitarás para la API

```txt
Ejemplo: 123-456-7890
```

## Paso 3: Configurar Proyecto en Google Cloud
1. Ve a [Google Cloud Console](https://console.cloud.google.com)
2. Crea un nuevo proyecto:
   - Haz clic en el selector de proyectos
   - "Nuevo proyecto"
   - Nombre: `Google Ads API Test`
   - Haz clic en "Crear"
3. Habilita Google Ads API:
   - Ve a "APIs y Servicios" > "Biblioteca"
   - Busca "Google Ads API"
   - Haz clic en "Habilitar"

## Paso 4: Crear Credenciales OAuth
1. En Cloud Console, ve a "APIs y Servicios" > "Credenciales"
2. Haz clic en "+ CREAR CREDENCIALES" > "ID de cliente OAuth"
3. Configura la pantalla de consentimiento:
   - Tipo de aplicación: "Aplicación de escritorio"
   - Nombre: `Google Ads API Access`
4. Haz clic en "Crear"
5. Descarga el archivo JSON de credenciales - lo necesitarás para autenticar

## Paso 5: Configurar Archivo `google-ads.yaml`
Crea un archivo de configuración con este formato:

```yaml
developer_token: "TU_DEVELOPER_TOKEN"
client_id: "TU_CLIENT_ID"
client_secret: "TU_CLIENT_SECRET"
refresh_token: "TU_REFRESH_TOKEN"
client_customer_id: "TU_CUSTOMER_ID"
use_proto_plus: true
```

Para obtener los valores:
- **developer_token**: Solicítalo en [Google Ads API Center](https://developers.google.com/google-ads/api/docs/first-call/dev-token)
- **client_id** y **client_secret**: Del archivo JSON descargado
- **refresh_token**: Generarlo con el script de Python (siguiente sección)
- **client_customer_id**: El ID de cliente obtenido en el Paso 2

## Paso 6: Generar Refresh Token (Python)
Ejecuta este script para obtener el refresh token:

```python
from google_auth_oauthlib.flow import InstalledAppFlow

# Configuración
CLIENT_SECRETS_FILE = "ruta/a/tu-archivo-credenciales.json"
SCOPES = ['https://www.googleapis.com/auth/adwords']

# Flujo de autenticación
flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
credentials = flow.run_local_server(port=0)

print("Refresh token:", credentials.refresh_token)
```

Guarda el refresh token generado en tu archivo `google-ads.yaml`.

## Paso 7: Probar la Conexión
Usa este script para verificar la conexión:

```python
from google.ads.googleads.client import GoogleAdsClient

client = GoogleAdsClient.load_from_storage("google-ads.yaml")
customer_service = client.get_service("CustomerService")
resource_name = customer_service.customer_path("TU_CUSTOMER_ID")
customer = customer_service.get_customer(resource_name=resource_name)

print(f"Cuenta verificada: {customer.descriptive_name}")
```

## Solución de Problemas Comunes
| Error | Solución |
|-------|----------|
| `PERMISSION_DENIED` | Verifica que el developer token sea válido |
| `UNAUTHENTICATED` | Renueva el refresh token |
| `CUSTOMER_NOT_FOUND` | Confirma el customer ID |
| `QUOTA_EXCEEDED` | Solicita mayor cuota en Cloud Console |

## Recursos Adicionales
1. [Documentación Oficial Google Ads API](https://developers.google.com/google-ads/api/docs/start)
2. [Ejemplos de Código](https://github.com/googleads/google-ads-python)
3. [Foro de Soporte](https://groups.google.com/g/adwords-api)
4. [Generar test account](https://developers.google.com/google-ads/api/docs/best-practices/test-accounts)
