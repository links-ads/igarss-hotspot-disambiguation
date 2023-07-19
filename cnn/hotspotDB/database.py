import logging
from pathlib import Path
import psycopg2
from psycopg2 import sql    
from config.schema import HotspotsDB

LOG = logging.getLogger(__name__)


class HotspotDatabase():
    def __init__(self, cfg:HotspotsDB) -> None:
        self.config = cfg
        self.db_connect()


    def db_connect(self):
        self.db = psycopg2.connect(host=self.config.host,
                                    database=self.config.database,
                                    user=self.config.user,
                                    password=self.config.password,
                                    port=self.config.port)
    
    def get_hotspot_data(self, table:str) -> list:
        try:
            if self.db.closed:
                self.db_connect()
            cursor = self.db.cursor()
            cursor.execute(sql.SQL("SELECT id, t.timestamp  AS hotspotDate, ST_AsText(t.firepixel) AS hotspotPoint FROM {table} as t").format(table=sql.Identifier(table)))
            data = cursor.fetchall()
            return data

        except Exception as e:
            LOG.error(f"Cannot perform query: {str(e)}")
            raise e
            
    def get_positive_hotspot(self, table:str) -> list:
        try:
            if self.db.closed:
                self.db_connect()
            cursor = self.db.cursor()
            cursor.execute(sql.SQL("SELECT id, t.timestamp  AS hotspotDate, ST_AsText(t.firepixel) AS hotspotPoint FROM {table} as t").format(table=sql.Identifier(table)))
            data = cursor.fetchall()
            return data

        except Exception as e:
            LOG.error(f"Cannot perform query: {str(e)}")
            raise e
            
    def get_hotspot(self, id):
        try:
            if self.db.closed:
                self.db_connect()
            cursor = self.db.cursor()
            cursor.execute(sql.SQL("SELECT unique_id, t.timestamp  AS hotspotDate, ST_AsText(t.firepixel) AS hotspotPoint FROM view_stratgrid_hotspot as t WHERE unique_id = %s"), id)
            data = cursor.fetchall()
            return data

        except Exception as e:
            LOG.error(f"Cannot perform query: {str(e)}")
            raise e
    
    def get_all_hotspot_geom(self):
        try:
            if self.db.closed:
                self.db_connect()
            cursor = self.db.cursor()
            cursor.execute(sql.SQL("SELECT unique_id, ST_AsText(t.firepixel) AS hotspotPoint, is_positive FROM view_stratgrid_hotspot as t"))
            data = cursor.fetchall()
            return data

        except Exception as e:
            LOG.error(f"Cannot perform query: {str(e)}")
            raise e
        
    def get_landcover_class(self, firepixel):
        try:
            if self.db.closed:
                self.db_connect()
            cursor = self.db.cursor()
            cursor.execute(sql.SQL("SELECT l.category FROM landcover as l where ST_Intersects(st_setsrid(ST_MakeValid(ST_GeomFromText(%s), 'method=structure'), 4326), l.shape)"), [firepixel])
            data = cursor.fetchall()
            return data

        except Exception as e:
            LOG.error(f"Cannot perform query: {str(e)}")
            raise e
