# ubicacion_finder.py
import googlemaps
import math
import pyproj
from shapely.geometry import Point, Polygon
import os
from dotenv import load_dotenv
import folium
import requests
import re
import geopy.distance
import logging

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorredorLocationFinder:
    def __init__(self, api_key=None):
        self.gmaps = googlemaps.Client(key=api_key or os.getenv('GOOGLE_MAPS_API_KEY'))
        self.geod = pyproj.Geod(ellps='WGS84')
    
    def obtener_viewport_distrito(self, distrito):
        """Obtiene el área rectangular que cubre el distrito"""
        try:
            resultado = self.gmaps.geocode(f"{distrito}, Lima, Perú")
            if not resultado:
                raise ValueError(f"No se encontró el distrito: {distrito}")
            
            viewport = resultado[0]['geometry']['viewport']
            logger.info(f"Viewport para {distrito}: {viewport}")
            return viewport
        except Exception as e:
            logger.error(f"Error al obtener viewport de {distrito}: {str(e)}")
            raise ValueError(f"Error al obtener área del distrito: {str(e)}")

    def punto_en_viewport(self, punto, viewport):
        """Verifica si un punto está dentro del viewport"""
        lat, lng = punto
        sw = viewport['southwest']
        ne = viewport['northeast']
        return (sw['lat'] <= lat <= ne['lat'] and 
                sw['lng'] <= lng <= ne['lng'])

    def buscar_referencia_en_area(self, referencia, area_viewport):
        """Busca una referencia dentro del área del distrito con verificación estricta"""
        try:
            # 1. Primero intentar con búsqueda restringida por componentes
            try:
                distrito = self.extraer_distrito_de_viewport(area_viewport)
                if distrito:
                    resultado = self.gmaps.geocode(
                        referencia,
                        components={
                            'administrative_area': 'Lima',
                            'locality': 'Lima',
                            'sublocality': distrito,
                            'country': 'PE'
                        }
                    )
                    if resultado:
                        location = resultado[0]['geometry']['location']
                        punto = (location['lat'], location['lng'])
                        if self.punto_en_viewport(punto, area_viewport):
                            return punto
            except Exception as e:
                logger.warning(f"Búsqueda por componentes falló, intentando con área: {str(e)}")

            # 2. Si falla, usar búsqueda por área con verificación
            sw = area_viewport['southwest']
            ne = area_viewport['northeast']
            center_lat = (sw['lat'] + ne['lat']) / 2
            center_lng = (sw['lng'] + ne['lng']) / 2
            
            # Calcular radio aproximado (en metros)
            radius = int(geopy.distance.distance(
                (sw['lat'], sw['lng']), 
                (ne['lat'], ne['lng'])
            ).m / 2)

            # Usar Places API para búsqueda en el área
            resultados = self.gmaps.places(
                query=referencia,
                location=(center_lat, center_lng),
                radius=radius
            )

            if resultados['status'] == 'OK' and resultados['results']:
                for result in resultados['results']:
                    location = result['geometry']['location']
                    punto = (location['lat'], location['lng'])
                    if self.punto_en_viewport(punto, area_viewport):
                        logger.info(f"Referencia encontrada en {punto}")
                        return punto
                
                raise ValueError("Se encontraron resultados pero ninguno está dentro del área del distrito")
            else:
                raise ValueError(f"No se encontró la referencia '{referencia}' en el área definida. Status: {resultados['status']}")
            
        except Exception as e:
            logger.error(f"Error al buscar referencia: {str(e)}")
            raise ValueError(f"Error al buscar la referencia en el área del distrito: {str(e)}")

    def extraer_distrito_de_viewport(self, viewport):
        """Intenta determinar el distrito a partir de las coordenadas del viewport"""
        center_lat = (viewport['southwest']['lat'] + viewport['northeast']['lat']) / 2
        center_lng = (viewport['southwest']['lng'] + viewport['northeast']['lng']) / 2
        
        reverse_geocode = self.gmaps.reverse_geocode((center_lat, center_lng))
        
        for component in reverse_geocode[0]['address_components']:
            if 'sublocality' in component['types']:
                return component['long_name']
        return None

    def buscar_lugar_cercano(self, keyword, referencia_coord):
        """Busca una dirección cerca de las coordenadas de referencia"""
        try:
            lat, lng = referencia_coord
            resultados = self.gmaps.places_nearby(
                location=(lat, lng),
                radius=1500,
                keyword=keyword
            )

            if resultados['status'] == 'OK' and resultados['results']:
                location = resultados['results'][0]['geometry']['location']
                logger.info(f"Dirección encontrada en {location}")
                return (location['lat'], location['lng'])
            else:
                raise ValueError(
                    f"No se encontró el lugar '{keyword}' cerca de {referencia_coord}. "
                    f"Status: {resultados['status']}"
                )
        except Exception as e:
            logger.error(f"Error al buscar dirección: {str(e)}")
            raise ValueError(f"Error al buscar la dirección: {str(e)}")

    def crear_corredor(self, inicio, fin, ancho_metros=100):
        """Crea un área rectangular entre dos puntos con un ancho específico"""
        # Convertir a coordenadas en radianes
        lat1, lon1 = math.radians(inicio[0]), math.radians(inicio[1])
        lat2, lon2 = math.radians(fin[0]), math.radians(fin[1])
        
        # Calcular azimut (dirección entre puntos)
        az = math.atan2(
            math.sin(lon2 - lon1) * math.cos(lat2),
            math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
        )
        
        # Calcular puntos perpendiculares
        az_perp1 = az + math.pi/2  # 90 grados
        az_perp2 = az - math.pi/2  # -90 grados
        
        # Calcular desplazamiento (ancho/2 en cada lado)
        desplazamiento = (ancho_metros/2) / 6371000  # Radio de la Tierra en metros
        
        # Puntos iniciales desplazados
        p1_lat = math.asin(math.sin(lat1) * math.cos(desplazamiento) + 
                 math.cos(lat1) * math.sin(desplazamiento) * math.cos(az_perp1))
        p1_lon = lon1 + math.atan2(
            math.sin(az_perp1) * math.sin(desplazamiento) * math.cos(lat1),
            math.cos(desplazamiento) - math.sin(lat1) * math.sin(p1_lat)
        )
        
        p2_lat = math.asin(math.sin(lat1) * math.cos(desplazamiento) + 
                 math.cos(lat1) * math.sin(desplazamiento) * math.cos(az_perp2))
        p2_lon = lon1 + math.atan2(
            math.sin(az_perp2) * math.sin(desplazamiento) * math.cos(lat1),
            math.cos(desplazamiento) - math.sin(lat1) * math.sin(p2_lat)
        )
        
        # Puntos finales desplazados
        p3_lat = math.asin(math.sin(lat2) * math.cos(desplazamiento) + 
                 math.cos(lat2) * math.sin(desplazamiento) * math.cos(az_perp2))
        p3_lon = lon2 + math.atan2(
            math.sin(az_perp2) * math.sin(desplazamiento) * math.cos(lat2),
            math.cos(desplazamiento) - math.sin(lat2) * math.sin(p3_lat)
        )
        
        p4_lat = math.asin(math.sin(lat2) * math.cos(desplazamiento) + 
                 math.cos(lat2) * math.sin(desplazamiento) * math.cos(az_perp1))
        p4_lon = lon2 + math.atan2(
            math.sin(az_perp1) * math.sin(desplazamiento) * math.cos(lat2),
            math.cos(desplazamiento) - math.sin(lat2) * math.sin(p4_lat)
        )
        
        # Convertir de vuelta a grados
        return [
            (math.degrees(p1_lat), math.degrees(p1_lon)),
            (math.degrees(p2_lat), math.degrees(p2_lon)),
            (math.degrees(p3_lat), math.degrees(p3_lon)),
            (math.degrees(p4_lat), math.degrees(p4_lon)),
            (math.degrees(p1_lat), math.degrees(p1_lon))  # Cerrar el polígono
        ]
    
    def calcular_punto_intermedio(self, inicio, fin, distancia_metros):
        """Calcula un punto a X metros de inicio en dirección a fin"""
        lon1, lat1 = inicio[1], inicio[0]
        lon2, lat2 = fin[1], fin[0]
        
        az, _, dist_total = self.geod.inv(lon1, lat1, lon2, lat2)
        
        if distancia_metros >= dist_total:
            return fin
        
        new_lon, new_lat, _ = self.geod.fwd(lon1, lat1, az, distancia_metros)
        return (new_lat, new_lon)
    
    def extraer_distrito(self, ubigeo: int) -> str:
        distritos_por_ubigeo = {
            150101: "Lima",
            150102: "Ancón",
            150103: "ATE",
            150104: "Barranco",
            150105: "Breña",
            150106: "Carabayllo",
            150107: "Chaclacayo",
            150108: "Chorrillos",
            150109: "Cieneguilla",
            150110: "Comas",
            150111: "El Agustino",
            150112: "Independencia",
            150113: "Jesús María",
            150114: "La Molina",
            150115: "La Victoria",
            150116: "Lince",
            150117: "Los Olivos",
            150118: "Lurigancho",
            150119: "Lurín",
            150120: "Magdalena del Mar",
            150121: "Pueblo Libre",
            150122: "Miraflores",
            150123: "Pachacamac",
            150124: "Pucusana",
            150125: "Puente Piedra",
            150126: "Punta Hermosa",
            150127: "Punta Negra",
            150128: "Rímac",
            150129: "San Bartolo",
            150130: "San Borja",
            150131: "San Isidro",
            150132: "San Juan de Lurigancho",
            150133: "San Juan de Miraflores",
            150134: "San Luis",
            150135: "San Martín de Porres",
            150136: "San Miguel",
            150137: "Santa Anita",
            150138: "Santa María del Mar",
            150139: "Santa Rosa",
            150140: "Santiago de Surco",
            150141: "Surquillo",
            150142: "Villa El Salvador",
            150143: "Villa María del Triunfo"
        }
        
        return distritos_por_ubigeo.get(ubigeo, "Lima")

    def mostrar_mapa(self, resultado):
        """Genera mapa interactivo con los resultados"""
        if resultado['status'] != 'success':
            raise ValueError("No se puede mostrar mapa para un resultado fallido")
        
        centro_mapa = [
            (resultado['coordenadas_referencia'][0] + resultado['coordenadas_direccion'][0])/2,
            (resultado['coordenadas_referencia'][1] + resultado['coordenadas_direccion'][1])/2
        ]
        
        m = folium.Map(location=centro_mapa, zoom_start=15)
        
        # Corredor rectangular
        folium.Polygon(
            locations=resultado['corredor'],
            color='blue',
            fill=True,
            fill_opacity=0.2,
            popup=f'Corredor de {resultado["ancho_corredor"]}m'
        ).add_to(m)
        
        # Línea central de referencia
        folium.PolyLine(
            locations=[resultado['coordenadas_referencia'], resultado['coordenadas_direccion']],
            color='black',
            weight=1,
            dash_array='5, 5',
            popup='Línea central'
        ).add_to(m)
        
        # Puntos clave
        folium.Marker(
            location=resultado['coordenadas_direccion'],
            icon=folium.Icon(color='blue', icon='home'),
            popup=f"Dirección\n{resultado['direccion']}"
        ).add_to(m)
        
        folium.Marker(
            location=resultado['coordenadas_referencia'],
            icon=folium.Icon(color='orange', icon='flag'),
            popup=f"Referencia\n{resultado['referencia']}"
        ).add_to(m)
        
        folium.Marker(
            location=resultado['punto_final'],
            icon=folium.Icon(color='red' if resultado['dentro_del_corredor'] else 'gray', icon='target'),
            popup=f"Punto calculado\n{resultado['punto_final']}\nDistancia: {resultado['distancia_metros']}m"
        ).add_to(m)
        
        # Línea de referencia a punto calculado
        folium.PolyLine(
            locations=[resultado['coordenadas_referencia'], resultado['punto_final']],
            color='green',
            weight=3,
            popup=f"Distancia medida: {resultado['distancia_metros']} metros"
        ).add_to(m)
        
        # Leyenda
        legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 200px; 
                        border:2px solid grey; z-index:9999; font-size:14px;
                        background-color:white; padding: 10px;">
                <b>Leyenda</b><br>
                <i class="fa fa-home" style="color:blue"></i> Dirección<br>
                <i class="fa fa-flag" style="color:orange"></i> Referencia<br>
                <i class="fa fa-bullseye" style="color:red"></i> Punto calculado<br>
                <div style="background-color:blue; opacity:0.2; width:100%; height:15px;"></div> Corredor<br>
                <div style="background-color:black; height:2px; width:100%;"></div> Línea central
            </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def encontrar_punto_final(self, direccion, referencia, distrito, distancia_metros, ancho_corredor=100):
        """Calcula el punto exacto a X metros de la referencia hacia la dirección dentro del corredor.
        Si no puede localizar la dirección exacta, usa las coordenadas de la referencia como punto base."""
        resultado_base = {
            'direccion': direccion,
            'referencia': referencia,
            'distrito': distrito,
            'distancia_metros': distancia_metros,
            'ancho_corredor': ancho_corredor
        }
        
        try:
            logger.info(f"Iniciando búsqueda para: {direccion} cerca de {referencia}")
            # 1. Obtener distrito y área de búsqueda
            distrito = self.extraer_distrito(distrito)

            if not distrito:
                logger.warning(f"No se pudo identificar el distrito en la dirección: {direccion}")
                return {
                    **resultado_base,
                    'status': 'no localizable',
                    'message': 'No se pudo identificar el distrito en la dirección',
                    'distrito': None
                }
            
            direccion_completa = direccion + " " + distrito
            coord_dist = self.obtener_viewport_distrito(distrito)
            if not coord_dist:
                logger.warning(f"No se pudo obtener coordenadas para el distrito: {distrito}")
                return {
                    **resultado_base,
                    'status': 'no localizable',
                    'message': 'No se pudo obtener coordenadas para el distrito',
                    'distrito': distrito
                }
            
            coord_ref = self.buscar_referencia_en_area(referencia, coord_dist)
            if not coord_ref:
                logger.warning(f"No se pudo encontrar la referencia: {referencia} en el área")
                return {
                    **resultado_base,
                    'status': 'no localizable',
                    'message': 'No se pudo encontrar la referencia en el área',
                    'distrito': distrito
                }
            
            coord_dir = self.buscar_lugar_cercano(direccion_completa, coord_ref)
            
            # Modificación clave: Si no encontramos la dirección, usamos la referencia como punto base
            usar_referencia_como_fallback = False
            if not coord_dir:
                logger.warning(f"No se pudo encontrar la dirección exacta: {direccion}. Usando referencia como fallback")
                coord_dir = coord_ref
                usar_referencia_como_fallback = True
            
            # 2. Calcular punto intermedio
            punto_final = self.calcular_punto_intermedio(coord_ref, coord_dir, distancia_metros)
            
            # 3. Crear y verificar corredor
            corredor = self.crear_corredor(coord_ref, coord_dir, ancho_corredor)
            poligono_corredor = Polygon(corredor)
            dentro_del_corredor = poligono_corredor.contains(Point(punto_final))
            
            logger.info("Búsqueda completada")
            
            return {
                **resultado_base,
                'status': 'success' if not usar_referencia_como_fallback else 'partial_success',
                'punto_final': punto_final,
                'coordenadas_direccion': coord_dir,
                'coordenadas_referencia': coord_ref,
                'dentro_del_corredor': dentro_del_corredor,
                'corredor': corredor,
                'distrito': distrito,
                'message': 'Dirección exacta no encontrada, usando referencia como punto base' if usar_referencia_como_fallback else None
            }
            
        except Exception as e:
            logger.error(f"Error inesperado en encontrar_punto_final: {str(e)}")
            return {
                **resultado_base,
                'status': 'no localizable',
                'message': f"Error inesperado: {str(e)}"
            }