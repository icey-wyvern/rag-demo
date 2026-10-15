\connect appdb

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- App role (idempotent)
DO $$
BEGIN
	IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_user') THEN
		CREATE ROLE app_user WITH LOGIN PASSWORD 'app_password' NOSUPERUSER NOCREATEDB NOCREATEROLE;
	END IF;
END$$;

-- Alembic runs as app_user and owns the application tables it creates.
GRANT USAGE, CREATE ON SCHEMA public TO app_user;
