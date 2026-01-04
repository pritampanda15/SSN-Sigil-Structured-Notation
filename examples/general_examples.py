"""
SSN General Examples

Demonstrates token-efficient LLM communication across various domains.
"""

import json
from ssn import SSN


def example_api_request():
    """REST API request specification."""
    
    ssn = SSN()
    
    ssn_text = """
@request|POST|/api/v1/users
>content_type:application/json
>auth:bearer
#retry
#validate
.body
  >name:John Doe
  >email:john@example.com
  >role:admin
.headers
  >x_request_id:abc123
  >x_client:mobile
"""
    
    json_data = {
        "method": "POST",
        "endpoint": "/api/v1/users",
        "content_type": "application/json",
        "auth": "bearer",
        "retry": True,
        "validate": True,
        "body": {
            "name": "John Doe",
            "email": "john@example.com",
            "role": "admin"
        },
        "headers": {
            "x_request_id": "abc123",
            "x_client": "mobile"
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("=== API Request Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_database_query():
    """Database query specification."""
    
    ssn = SSN()
    
    ssn_text = """
@query|SELECT|users
>fields:id,name,email,created_at
>where:status=active AND role=admin
>order:created_at DESC
>limit:50
>offset:0
#distinct
.join
  >table:orders
  >on:users.id=orders.user_id
  >type:LEFT
"""
    
    json_data = {
        "action": "SELECT",
        "table": "users",
        "fields": ["id", "name", "email", "created_at"],
        "where": "status=active AND role=admin",
        "order": "created_at DESC",
        "limit": 50,
        "offset": 0,
        "distinct": True,
        "join": {
            "table": "orders",
            "on": "users.id=orders.user_id",
            "type": "LEFT"
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Database Query Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_ml_training():
    """Machine learning training configuration."""
    
    ssn = SSN()
    
    ssn_text = """
@train|transformer|sentiment_model
>epochs:100
>batch_size:32
>lr:0.0001
>optimizer:adamw
>scheduler:cosine
#mixed_precision
#gradient_checkpointing
.data
  >train:data/train.csv
  >val:data/val.csv
  >test:data/test.csv
  >max_len:512
.model
  >base:bert-base-uncased
  >num_labels:3
  >dropout:0.1
.callbacks
  #early_stopping
  #model_checkpoint
  >patience:5
"""
    
    json_data = {
        "task": "train",
        "model_type": "transformer",
        "model_name": "sentiment_model",
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "mixed_precision": True,
        "gradient_checkpointing": True,
        "data": {
            "train": "data/train.csv",
            "val": "data/val.csv",
            "test": "data/test.csv",
            "max_len": 512
        },
        "model": {
            "base": "bert-base-uncased",
            "num_labels": 3,
            "dropout": 0.1
        },
        "callbacks": {
            "early_stopping": True,
            "model_checkpoint": True,
            "patience": 5
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== ML Training Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_ci_cd_pipeline():
    """CI/CD pipeline configuration."""
    
    ssn = SSN()
    
    ssn_text = """
@pipeline|deploy_prod
>trigger:push
>branch:main
#parallel
.stage|build
  @run|npm install
  @run|npm run build
  >timeout:10m
  #cache_deps
.stage|test
  @run|npm test
  @run|npm run lint
  #coverage
  >min_coverage:80
.stage|deploy
  @run|aws s3 sync
  >bucket:prod-assets
  >region:us-east-1
  #invalidate_cdn
"""
    
    json_data = {
        "pipeline": "deploy_prod",
        "trigger": "push",
        "branch": "main",
        "parallel": True,
        "stages": [
            {
                "name": "build",
                "steps": ["npm install", "npm run build"],
                "timeout": "10m",
                "cache_deps": True
            },
            {
                "name": "test",
                "steps": ["npm test", "npm run lint"],
                "coverage": True,
                "min_coverage": 80
            },
            {
                "name": "deploy",
                "steps": ["aws s3 sync"],
                "bucket": "prod-assets",
                "region": "us-east-1",
                "invalidate_cdn": True
            }
        ]
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== CI/CD Pipeline Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_data_transformation():
    """Data transformation/ETL specification."""
    
    ssn = SSN()
    
    ssn_text = """
@etl|customer_analytics
.extract
  >source:postgres
  >table:raw_events
  >query:SELECT * WHERE date > '2024-01-01'
.transform
  @filter|event_type IN ('purchase','signup')
  @groupby|user_id,date
  @agg|sum:amount;count:events;avg:session_time
  @rename|amount_sum:total_revenue
  #drop_nulls
.load
  >dest:snowflake
  >table:analytics.customer_metrics
  >mode:upsert
  >key:user_id,date
"""
    
    json_data = {
        "pipeline": "customer_analytics",
        "extract": {
            "source": "postgres",
            "table": "raw_events",
            "query": "SELECT * WHERE date > '2024-01-01'"
        },
        "transform": {
            "filter": "event_type IN ('purchase','signup')",
            "groupby": ["user_id", "date"],
            "aggregations": {
                "sum": "amount",
                "count": "events",
                "avg": "session_time"
            },
            "rename": {"amount_sum": "total_revenue"},
            "drop_nulls": True
        },
        "load": {
            "destination": "snowflake",
            "table": "analytics.customer_metrics",
            "mode": "upsert",
            "key": ["user_id", "date"]
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Data Transformation Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_chatbot_config():
    """Chatbot/Agent configuration."""
    
    ssn = SSN()
    
    ssn_text = """
@agent|customer_support
>model:gpt-4
>temperature:0.7
>max_tokens:1000
#memory
#tools
.persona
  >name:Alex
  >tone:friendly,professional
  >language:en
.tools
  >search:knowledge_base
  >action:create_ticket
  >action:check_order
  >action:process_refund
.guardrails
  #no_pii
  #escalate_angry
  >max_turns:10
"""
    
    json_data = {
        "agent": "customer_support",
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000,
        "memory": True,
        "tools": True,
        "persona": {
            "name": "Alex",
            "tone": ["friendly", "professional"],
            "language": "en"
        },
        "available_tools": {
            "search": "knowledge_base",
            "actions": ["create_ticket", "check_order", "process_refund"]
        },
        "guardrails": {
            "no_pii": True,
            "escalate_angry": True,
            "max_turns": 10
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Chatbot Config Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_image_generation():
    """Image generation prompt specification."""
    
    ssn = SSN()
    
    ssn_text = """
@generate|image
>prompt:A serene mountain landscape at sunset
>negative:blurry,low quality,watermark
>model:sdxl
>size:1024x1024
>steps:30
>cfg:7.5
>seed:42
#highres_fix
.controlnet
  >type:canny
  >strength:0.8
  >image:reference.png
"""
    
    json_data = {
        "task": "generate",
        "type": "image",
        "prompt": "A serene mountain landscape at sunset",
        "negative_prompt": "blurry,low quality,watermark",
        "model": "sdxl",
        "size": "1024x1024",
        "steps": 30,
        "cfg_scale": 7.5,
        "seed": 42,
        "highres_fix": True,
        "controlnet": {
            "type": "canny",
            "strength": 0.8,
            "image": "reference.png"
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Image Generation Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_task_scheduling():
    """Task/job scheduling specification."""
    
    ssn = SSN()
    
    ssn_text = """
@schedule|daily_report
>cron:0 8 * * *
>timezone:America/New_York
#enabled
#notify_on_failure
.task
  @run|python generate_report.py
  >args:--date yesterday --format pdf
  >timeout:30m
  >retries:3
.notify
  >email:team@company.com
  >slack:#reports
  #attach_output
"""
    
    json_data = {
        "schedule": "daily_report",
        "cron": "0 8 * * *",
        "timezone": "America/New_York",
        "enabled": True,
        "notify_on_failure": True,
        "task": {
            "command": "python generate_report.py",
            "args": "--date yesterday --format pdf",
            "timeout": "30m",
            "retries": 3
        },
        "notify": {
            "email": "team@company.com",
            "slack": "#reports",
            "attach_output": True
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Task Scheduling Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_kubernetes_deployment():
    """Kubernetes deployment specification."""
    
    ssn = SSN()
    
    ssn_text = """
@deploy|web-app
>namespace:production
>replicas:3
>image:myapp:v1.2.3
.resources
  >cpu_req:100m
  >cpu_lim:500m
  >mem_req:128Mi
  >mem_lim:512Mi
.env
  >DATABASE_URL:secret:db-creds
  >REDIS_HOST:redis-svc
  >LOG_LEVEL:info
.probes
  >liveness:/health
  >readiness:/ready
  >port:8080
#rolling_update
>max_surge:1
>max_unavailable:0
"""
    
    json_data = {
        "deployment": "web-app",
        "namespace": "production",
        "replicas": 3,
        "image": "myapp:v1.2.3",
        "resources": {
            "requests": {"cpu": "100m", "memory": "128Mi"},
            "limits": {"cpu": "500m", "memory": "512Mi"}
        },
        "env": {
            "DATABASE_URL": {"secretRef": "db-creds"},
            "REDIS_HOST": "redis-svc",
            "LOG_LEVEL": "info"
        },
        "probes": {
            "liveness": "/health",
            "readiness": "/ready",
            "port": 8080
        },
        "strategy": {
            "type": "rolling_update",
            "max_surge": 1,
            "max_unavailable": 0
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Kubernetes Deployment Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_web_scraping():
    """Web scraping task specification."""
    
    ssn = SSN()
    
    ssn_text = """
@scrape|product_prices
>url:https://example.com/products
>selector:.product-card
#paginate
#javascript
.extract
  >title:.product-title
  >price:.price-value
  >rating:.star-rating@data-rating
  >image:img@src
.options
  >delay:2
  >max_pages:10
  >proxy:rotating
  >user_agent:random
#save_html
>output:products.json
"""
    
    json_data = {
        "task": "scrape",
        "name": "product_prices",
        "url": "https://example.com/products",
        "selector": ".product-card",
        "paginate": True,
        "javascript": True,
        "extract": {
            "title": ".product-title",
            "price": ".price-value",
            "rating": {"selector": ".star-rating", "attr": "data-rating"},
            "image": {"selector": "img", "attr": "src"}
        },
        "options": {
            "delay": 2,
            "max_pages": 10,
            "proxy": "rotating",
            "user_agent": "random"
        },
        "save_html": True,
        "output": "products.json"
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Web Scraping Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


if __name__ == "__main__":
    example_api_request()
    example_database_query()
    example_ml_training()
    example_ci_cd_pipeline()
    example_data_transformation()
    example_chatbot_config()
    example_image_generation()
    example_task_scheduling()
    example_kubernetes_deployment()
    example_web_scraping()
