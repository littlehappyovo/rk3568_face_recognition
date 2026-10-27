"""Authenticated business API. Never exposes SQL, shell commands or file paths."""

import base64
import csv
from datetime import datetime, timezone
from functools import wraps
import hmac
import io
import ipaddress
from urllib.parse import urlsplit

from flask import Flask, jsonify, request, Response, render_template
from werkzeug.exceptions import HTTPException

from .auth import AuthError, AuthManager
from .config import ServerConfig
from .images import checked_jpeg
from .jobs import AccessJobs
from .qt_bridge import BridgeError, QtBridge
from .store import AccessStore, StoreConflict, StoreNotFound, StoreValidationError, _text


def create_app(config: ServerConfig, bridge: QtBridge) -> Flask:
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=8 * 1024 * 1024 + 65536)
    store = AccessStore(config.runtime_dir / "access.sqlite3")
    auth, jobs = AuthManager(store), AccessJobs(bridge)
    app.extensions.update(access_store=store, access_auth=auth, access_jobs=jobs)
    def payload():
        if not request.is_json:
            raise StoreValidationError("请求必须使用JSON")
        value = request.get_json()
        if not isinstance(value, dict):
            raise StoreValidationError("请求内容必须是对象")
        return value

    def same_origin():
        origin = request.headers.get("Origin")
        expected = urlsplit(request.host_url)
        supplied = urlsplit(origin or "")
        if not origin or (supplied.scheme, supplied.netloc, supplied.path) != (expected.scheme, expected.netloc, ""):
            raise AuthError("请求来源校验失败", 403)

    def administrator(write=False):
        def decorate(function):
            @wraps(function)
            def checked(*args, **kwargs):
                csrf = None
                if write:
                    same_origin()
                    csrf = request.headers.get("X-CSRF-Token", "")
                auth.authorize(request.cookies.get("face_admin"), csrf=csrf)
                return function(*args, **kwargs)
            return checked
        return decorate

    def local_only():
        try:
            loopback = ipaddress.ip_address(request.remote_addr).is_loopback
        except ValueError:
            loopback = False
        supplied = request.headers.get("X-Local-Secret", "")
        if not loopback or len(supplied) > 128 or not supplied.isascii() or not hmac.compare_digest(supplied, config.local_secret):
            raise AuthError("仅限板端本机操作", 403)

    def identifier(value):
        _text(value, "ID", 128)
        if value in (".", "..") or any(char in value for char in ("/", "\\", "\r", "\n")):
            raise StoreValidationError("ID格式错误")
        return value

    def listed(method):
        args = request.args.to_dict()
        try:
            page, limit = int(args.pop("page", "1")), int(args.pop("limit", "50"))
        except ValueError as error:
            raise StoreValidationError("分页参数错误") from error
        for key in ("include_history", "resolved"):
            if key in args:
                if args[key] not in ("true", "false"):
                    raise StoreValidationError("筛选状态错误")
                args[key] = args[key] == "true"
        store.expire_pending(datetime.now(timezone.utc))
        return method(args, page=page, limit=limit)

    @app.after_request
    def response_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.errorhandler(Exception)
    def api_error(error):
        status, code, message = 500, "internal_error", "后台操作失败"
        if isinstance(error, AuthError):
            status, code, message = error.status, "authentication_failed", str(error)
        elif isinstance(error, StoreValidationError):
            status, code, message = 400, "invalid_input", str(error)
        elif isinstance(error, StoreConflict):
            status, code, message = 409, "conflict", str(error)
        elif isinstance(error, StoreNotFound):
            status, code, message = 404, "not_found", str(error)
        elif isinstance(error, BridgeError):
            status, code, message = 503, "board_unavailable", str(error)
        elif isinstance(error, HTTPException):
            status, code, message = error.code, error.name.lower().replace(" ", "_"), error.description
        return jsonify(error=dict(code=code, message=message)), status

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/auth/status")
    def auth_status():
        return jsonify(initialized=auth.initialized())

    @app.post("/api/local/setup")
    def setup():
        local_only()
        auth.setup(payload().get("password"))
        return jsonify(initialized=True), 201

    @app.post("/api/auth/login")
    def login():
        if request.headers.get("Origin"):
            same_origin()
        session = auth.login(payload().get("password"), request.remote_addr)
        response = jsonify(csrf_token=session["csrf_token"], expires_at=session["expires_at"])
        response.set_cookie("face_admin", session["token"], httponly=True, samesite="Strict", path="/", max_age=28800)
        return response

    @app.post("/api/auth/session")
    @app.get("/api/auth/session")
    @administrator(write=True)
    def session_activity():
        return jsonify(auth.authorize(request.cookies.get("face_admin"),
                                     csrf=request.headers.get("X-CSRF-Token"), refresh=True))

    @app.get("/api/auth/check")
    @administrator()
    def check_session():
        return jsonify(auth.authorize(request.cookies.get("face_admin")))

    @app.post("/api/auth/logout")
    @administrator(write=True)
    def logout():
        auth.logout(request.cookies.get("face_admin"))
        response = jsonify(logged_out=True)
        response.delete_cookie("face_admin", path="/")
        return response

    @app.post("/api/auth/password")
    @administrator(write=True)
    def change_password():
        values = payload()
        auth.change_password(request.cookies.get("face_admin"), values.get("current_password"), values.get("new_password"))
        response = jsonify(changed=True)
        response.delete_cookie("face_admin", path="/")
        return response

    @app.post("/api/local/events")
    def local_events():
        local_only()
        return jsonify(store.ingest(payload()))

    @app.get("/api/events")
    @administrator()
    def events():
        return jsonify(listed(store.list_events))

    @app.get("/api/visits")
    @administrator()
    def visits():
        return jsonify(listed(store.list_visits))

    @app.get("/api/anomalies")
    @administrator()
    def anomalies():
        return jsonify(listed(store.list_anomalies))

    @app.post("/api/anomalies/<anomaly_id>/resolve")
    @administrator(write=True)
    def resolve(anomaly_id):
        return jsonify(store.resolve_anomaly(anomaly_id, payload().get("note"), "admin"))

    @app.get("/api/events.csv")
    @administrator()
    def export_csv():
        args = request.args.to_dict()
        args.pop("page", None)
        args.pop("limit", None)
        store.expire_pending(datetime.now(timezone.utc))
        rows = store.export_events(args)
        output = io.StringIO()
        writer = csv.writer(output)
        fields = ("event_id", "observed_at", "station_id", "direction", "person_id", "display_name", "decision")
        writer.writerow(fields)
        for row in rows:
            values = [str(row[key] if row[key] is not None else "") for key in fields]
            writer.writerow(["'" + value if value.lstrip(" \t\r\n").startswith(("=", "+", "-", "@"))
                             or value.startswith(("\t", "\r", "\n")) else value for value in values])
        return Response("\ufeff" + output.getvalue(), mimetype="text/csv", headers={
            "Content-Disposition": 'attachment; filename="recognition-events.csv"',
            "X-Export-Limit": "10000"})

    @app.get("/api/stations")
    @administrator()
    def stations():
        return jsonify(items=store.list_stations(), settings=store.settings())

    @app.patch("/api/stations/<station_id>")
    @administrator(write=True)
    def configure_station(station_id):
        if station_id != "board-in":
            raise StoreNotFound("终端不存在")
        station = store.update_station(identifier(station_id), payload())
        bridge.call("station.configure", station)
        return jsonify(station)

    @app.patch("/api/settings")
    @administrator(write=True)
    def settings():
        return jsonify(store.update_settings(payload()))

    @app.get("/api/local/stations")
    def local_stations():
        local_only()
        return jsonify(items=store.list_stations())

    @app.get("/api/people")
    @administrator()
    def people():
        return jsonify(bridge.call("people.list", request.args.to_dict()))

    def validate_person(values, create=False):
        allowed = {"display_name", "group", "enabled"} | ({"person_id"} if create else set())
        if not values or set(values) - allowed:
            raise StoreValidationError("人员字段错误")
        if create:
            identifier(values.get("person_id"))
        if "display_name" in values:
            _text(values["display_name"], "姓名", 128)
        elif create:
            raise StoreValidationError("缺少姓名")
        if "enabled" in values and type(values["enabled"]) is not bool:
            raise StoreValidationError("启用状态错误")
        if "group" in values and values["group"] not in ("in_group", "out_group"):
            raise StoreValidationError("组别错误")
        return values

    @app.post("/api/people")
    @administrator(write=True)
    def create_person():
        return jsonify(jobs.submit("people.create", validate_person(payload(), create=True))), 202

    @app.patch("/api/people/<person_id>")
    @administrator(write=True)
    def update_person(person_id):
        values = validate_person(payload())
        return jsonify(jobs.submit("people.update", dict(values, person_id=identifier(person_id)))), 202

    @app.delete("/api/people/<person_id>")
    @administrator(write=True)
    def delete_person(person_id):
        return jsonify(jobs.submit("people.delete", dict(person_id=identifier(person_id)))), 202

    @app.get("/api/people/<person_id>/samples")
    @administrator()
    def samples(person_id):
        return jsonify(bridge.call("samples.list", dict(request.args, person_id=identifier(person_id))))

    @app.post("/api/people/<person_id>/samples")
    @administrator(write=True)
    def add_sample(person_id):
        identifier(person_id)
        if "image" not in request.files:
            raise StoreValidationError("缺少照片")
        jpeg = checked_jpeg(request.files["image"].stream.read(8 * 1024 * 1024 + 1))
        return jsonify(jobs.submit("samples.add", dict(person_id=person_id,
                            jpeg_base64=base64.b64encode(jpeg).decode("ascii")))), 202

    @app.get("/api/people/<person_id>/samples/<sample_id>")
    @administrator()
    def read_sample(person_id, sample_id):
        result = bridge.call("samples.read", dict(person_id=identifier(person_id), sample_id=identifier(sample_id)))
        try:
            data = base64.b64decode(result["jpeg_base64"], validate=True)
        except (KeyError, ValueError, TypeError) as error:
            raise BridgeError("板端照片响应无效") from error
        if len(data) > 512 * 1024:
            raise BridgeError("板端缩略图过大")
        return Response(data, mimetype="image/jpeg")

    @app.delete("/api/people/<person_id>/samples/<sample_id>")
    @administrator(write=True)
    def delete_sample(person_id, sample_id):
        return jsonify(jobs.submit("samples.delete", dict(person_id=identifier(person_id), sample_id=identifier(sample_id)))), 202

    @app.post("/api/gallery/rebuild")
    @administrator(write=True)
    def rebuild():
        return jsonify(jobs.submit("gallery.rebuild", {})), 202

    @app.get("/api/jobs/<job_id>")
    @administrator()
    def job(job_id):
        return jsonify(jobs.get(identifier(job_id)))

    return app
