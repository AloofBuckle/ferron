use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use ferron_core::pipeline::Pipeline;
use ferron_http::span::HttpContextSpanExt;
use ferron_http::{trace_context, HttpContext, HttpFileContext, HttpResponse};
use ferron_observability::{
    CompositeEventSink, Event, LogAttributeValue, Parent, TraceAttributeValue, TraceEvent,
};

use super::observability::PerStageSpanHooks;

use super::file_pipeline::{
    execute_http_file_pipeline, strip_matched_path_prefix, FilePipelineExecutionError,
};
use super::request_utils::emit_error_with_trace;

#[allow(clippy::too_many_arguments)]
pub async fn execute_pipeline_stages(
    ctx: &mut HttpContext,
    pipeline: &Pipeline<HttpContext>,
    file_pipeline: &Pipeline<HttpFileContext>,
    events: &CompositeEventSink,
    log_prefix: &str,
    path_segments: &[String],
    request_span_key: Option<&str>,
    timeout_duration: Option<Duration>,
    control_plane_metadata: Option<Arc<std::collections::BTreeMap<String, String>>>,
) {
    let has_traces = events.has_trace_sinks();
    let pipeline_span_key =
        request_span_key.map(|_| super::observability::next_span_key("pipeline"));
    let log_trace_context = ctx
        .get::<trace_context::TraceContextKey>()
        .map(trace_context::to_event_trace_context);

    // Remove the base URL if path segments were matched
    if !path_segments.is_empty() {
        if let Some(req) = ctx.req.take() {
            let (mut parts, body) = req.into_parts();
            let mut uri_parts = parts.uri.into_parts();
            if let Some(path_and_query) = uri_parts.path_and_query {
                uri_parts.path_and_query =
                    strip_matched_path_prefix(&path_and_query, path_segments.len());
                if uri_parts.path_and_query.is_none() {
                    ctx.res = Some(HttpResponse::BuiltinError(400, None));
                    return;
                }
            }
            let Ok(new_uri) = http::Uri::from_parts(uri_parts) else {
                ctx.res = Some(HttpResponse::BuiltinError(400, None));
                return;
            };
            parts.uri = new_uri;
            ctx.req = Some(http::Request::from_parts(parts, body));
        }
    }

    if let (true, Some(request_span_key), Some(pipeline_span_key)) =
        (has_traces, request_span_key, pipeline_span_key.as_ref())
    {
        events.emit(Event::Trace(TraceEvent::StartSpan {
            key: Cow::Owned(pipeline_span_key.clone()),
            name: Cow::Borrowed("ferron.pipeline.execute"),
            parent: Some(Parent::ByKey(request_span_key.to_string())),
            trace_context: None,
            builder_attributes: vec![],
            attributes: vec![(
                "ferron.pipeline.log_prefix",
                TraceAttributeValue::String(log_prefix.to_string()),
            )],
            links: vec![],
            control_plane_metadata: control_plane_metadata.clone(),
        }));
    }

    let instant = std::time::Instant::now();

    let mut stage_hooks = PerStageSpanHooks::new(
        events,
        has_traces && pipeline_span_key.is_some(),
        pipeline_span_key.as_deref().unwrap_or(""),
        "http",
        control_plane_metadata.clone(),
    );

    let mut executed_stages = Vec::new();
    let forward_completed = match if let Some(timeout_duration) =
        timeout_duration.map(|d| d.saturating_sub(instant.elapsed()))
    {
        zincio::time::timeout(
            timeout_duration,
            pipeline.execute_forward_with_hooks(ctx, &mut stage_hooks, &mut executed_stages),
        )
        .await
    } else {
        Ok(pipeline
            .execute_forward_with_hooks(ctx, &mut stage_hooks, &mut executed_stages)
            .await)
    } {
        Ok(Ok(())) => true,
        Ok(Err(error)) => {
            emit_error_with_trace(
                events,
                format!("{log_prefix}Pipeline execution error: {error}"),
                log_trace_context.clone(),
                vec![
                    (
                        "error.type",
                        LogAttributeValue::String("pipeline_error".into()),
                    ),
                    (
                        "error.message",
                        LogAttributeValue::String(error.to_string()),
                    ),
                ],
            );
            ctx.res = Some(HttpResponse::BuiltinError(500, None));
            false
        }
        Err(_) => {
            // The stage future was cancelled, so `after_stage` never ran and
            // the in-flight span would otherwise end with no attributes.
            // Record which timeout fired, then attach anything staged before
            // the timeout (e.g. proxy backend selection) to that span.
            if let Some(timeout) = timeout_duration {
                ctx.get_span_attributes().insert(
                    "ferron.pipeline.timeout_secs",
                    TraceAttributeValue::F64(timeout.as_secs_f64()),
                );
            }
            stage_hooks.flush_with_context(ctx);
            emit_error_with_trace(
                events,
                format!("{log_prefix}Pipeline execution timeout"),
                log_trace_context.clone(),
                vec![(
                    "error.type",
                    LogAttributeValue::String("pipeline_timeout".into()),
                )],
            );
            ctx.res = Some(HttpResponse::BuiltinError(408, None));
            false
        }
    };

    if forward_completed && ctx.res.is_none() {
        match execute_http_file_pipeline(
            ctx,
            file_pipeline,
            timeout_duration.map(|d| d.saturating_sub(instant.elapsed())),
            pipeline_span_key.as_deref(),
            control_plane_metadata.clone(),
        )
        .await
        {
            Ok(()) => {}
            Err(FilePipelineExecutionError::Forbidden { .. }) => {
                ctx.res = Some(HttpResponse::BuiltinError(403, None));
            }
            Err(FilePipelineExecutionError::BadRequest { .. }) => {
                ctx.res = Some(HttpResponse::BuiltinError(400, None));
            }
            Err(FilePipelineExecutionError::Timeout) => {
                ctx.res = Some(HttpResponse::BuiltinError(408, None));
            }
            Err(FilePipelineExecutionError::Io(error)) => {
                emit_error_with_trace(
                    events,
                    format!("{log_prefix}HTTP file resolution error: {error}"),
                    log_trace_context.clone(),
                    vec![
                        (
                            "error.type",
                            LogAttributeValue::String("file_resolution_error".into()),
                        ),
                        (
                            "error.message",
                            LogAttributeValue::String(error.to_string()),
                        ),
                    ],
                );
                ctx.res = Some(HttpResponse::BuiltinError(500, None));
            }
            Err(FilePipelineExecutionError::Pipeline(error)) => {
                emit_error_with_trace(
                    events,
                    format!("{log_prefix}Pipeline execution error: {error}"),
                    log_trace_context.clone(),
                    vec![
                        (
                            "error.type",
                            LogAttributeValue::String("pipeline_error".into()),
                        ),
                        (
                            "error.message",
                            LogAttributeValue::String(error.to_string()),
                        ),
                    ],
                );
                ctx.res = Some(HttpResponse::BuiltinError(500, None));
            }
        }
    }

    if let Err(error) = pipeline
        .execute_inverse_with_hooks(ctx, executed_stages, &mut stage_hooks)
        .await
    {
        emit_error_with_trace(
            events,
            format!("{log_prefix}Pipeline inverse execution error: {error}"),
            log_trace_context,
            vec![(
                "error.type",
                LogAttributeValue::String("pipeline_inverse_error".into()),
            )],
        );
        ctx.res = Some(HttpResponse::BuiltinError(500, None));
    }

    // Flush any remaining stage spans before ending the pipeline execution span.
    drop(stage_hooks);

    // End pipeline execution span
    if let Some(pipeline_span_key) = pipeline_span_key {
        events.emit(Event::Trace(TraceEvent::EndSpan {
            key: Cow::Owned(pipeline_span_key),
            name: Cow::Borrowed("ferron.pipeline.execute"),
            error: ctx.res.as_ref().and_then(|r| match r {
                HttpResponse::BuiltinError(s, _) if *s >= 400 => {
                    Some(format!("builtin error {}", s))
                }
                _ => None,
            }),
            attributes: vec![],
            control_plane_metadata: control_plane_metadata.clone(),
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use ferron_core::pipeline::{PipelineError, Stage};
    use std::{
        future::pending,
        sync::atomic::{AtomicUsize, Ordering},
    };

    struct CompletedStage(Arc<AtomicUsize>);

    #[async_trait(?Send)]
    impl Stage<HttpContext> for CompletedStage {
        fn name(&self) -> &str {
            "completed"
        }

        async fn run(&self, _ctx: &mut HttpContext) -> Result<bool, PipelineError> {
            Ok(true)
        }

        async fn run_inverse(&self, _ctx: &mut HttpContext) -> Result<(), PipelineError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    struct InterruptedStage {
        pending: bool,
    }

    #[async_trait(?Send)]
    impl Stage<HttpContext> for InterruptedStage {
        fn name(&self) -> &str {
            "interrupted"
        }

        async fn run(&self, _ctx: &mut HttpContext) -> Result<bool, PipelineError> {
            if self.pending {
                pending().await
            } else {
                Err(PipelineError::custom("forward failed"))
            }
        }
    }

    async fn execute_interrupted_pipeline(pending: bool) -> (HttpContext, usize) {
        let inverse_calls = Arc::new(AtomicUsize::new(0));
        let pipeline = Pipeline::new()
            .add_stage(Arc::new(CompletedStage(Arc::clone(&inverse_calls))))
            .add_stage(Arc::new(InterruptedStage { pending }));
        let file_pipeline = Pipeline::<HttpFileContext>::new();
        let mut ctx = HttpContext::default();
        let events = CompositeEventSink::default();
        execute_pipeline_stages(
            &mut ctx,
            &pipeline,
            &file_pipeline,
            &events,
            "",
            &[],
            None,
            pending.then_some(Duration::from_millis(10)),
            None,
        )
        .await;
        (ctx, inverse_calls.load(Ordering::SeqCst))
    }

    #[tokio::test(flavor = "current_thread")]
    async fn forward_error_runs_inverse_before_returning_500() {
        let (ctx, inverse_calls) = execute_interrupted_pipeline(false).await;
        assert_eq!(inverse_calls, 1);
        assert!(matches!(ctx.res, Some(HttpResponse::BuiltinError(500, _))));
    }

    #[test]
    fn forward_timeout_runs_inverse_before_returning_408() {
        let runtime = zincio::RuntimeBuilder::new()
            .driver(zincio::DriverKind::Mock)
            .enable_timer(true)
            .build()
            .expect("failed to build zincio test runtime");
        let (ctx, inverse_calls) = runtime.block_on(execute_interrupted_pipeline(true));
        assert_eq!(inverse_calls, 1);
        assert!(matches!(ctx.res, Some(HttpResponse::BuiltinError(408, _))));
    }
}
