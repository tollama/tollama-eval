{{/*
Expand the name of the chart.
*/}}
{{- define "tollama-eval.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "tollama-eval.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "tollama-eval.labels" -}}
helm.sh/chart: {{ include "tollama-eval.name" . }}-{{ .Chart.Version }}
{{ include "tollama-eval.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "tollama-eval.selectorLabels" -}}
app.kubernetes.io/name: {{ include "tollama-eval.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
